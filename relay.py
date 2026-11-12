"""Baton token-relay helpers: resume a half-finished generation on another worker.

The text a model has emitted so far is a hardware-independent checkpoint. Workers stream it to
the coordinator on the heartbeat they already send (as deltas), and with the final preempt
notice; when a lease ends without a completion (human returned, Wi-Fi drop, SIGKILL), the next
worker on the same model continues the generation from that text instead of from token zero.
(Design and live measurements: hackathon/t2-fable "Baton", with guards from t5-opus "Tidal".)

Resuming needs the model's prompt rendered with the assistant turn left open and the prefix
already "said". Ollama's chat endpoint can't do that reliably (measured: it loses its place and
drifts into prose), so we render the prompt ourselves and use `/api/generate` with `raw: true`.
That is only correct for a template we can reproduce byte-for-byte - today ChatML (Qwen) - and the
worker *verifies* it at startup by comparing prompt token counts with `/api/chat`
(`ChatML.verify`). A model that fails the check simply restarts chunks, as before.

Resume ladder (worker side):
  1. naive continuation of the prefix, ungrammared (Ollama's `format` grammar always restarts at
     the root), watched by an online JSON validator and stopped as soon as the object closes;
  2. on a syntax slip: roll back to the last complete array item and decode each remaining item
     under its own JSON-schema grammar, emitting the structural glue ourselves;
  3. if that fails validation too: /worker/fail -> the coordinator drops the checkpoint, so a bad
     prefix can never loop, and the chunk restarts from zero.
"""
from __future__ import annotations

import json
import re
from typing import Awaitable, Callable, Optional

MAX_CKPT_CHARS = 64_000      # coordinator-side cap on a stored checkpoint
MAX_RESUME_TOKENS = 1024     # cap on any continuation: a runaway resume must never hog a node
MAX_ITEMS = 48               # cap on grammar-resumed array items
# A checkpoint is untrusted input to the *next* worker's raw prompt: special tokens in it could close
# the assistant turn and inject a system/user message into another node's model. Allowlist, not a
# denylist of known tokens: checkpoints may not contain '<' or '>' at all (extraction JSON never needs
# them raw), and a chunk is only resumed if it has no tag-like '<x', '</', '<|' or '<!' sequence.
_TAG_LIKE = re.compile(r"<[A-Za-z/|!]")


def safe_for_raw_prompt(text: str) -> bool:
    """For the chunk (user) text rendered into a resume prompt."""
    return _TAG_LIKE.search(text) is None


def validated_checkpoint(text: str) -> Optional[str]:
    """What a coordinator stores for `text`: its validated extraction-JSON prefix (ends at the last
    complete array item, so no unvalidated tail and nothing after the object closes), or None if
    the text isn't an extraction prefix or contains any angle bracket."""
    if not text.lstrip().startswith("{") or "<" in text or ">" in text:
        return None
    good = closable(text)
    return good or None


# --------------------------------------------------------------------------- JSON progress
def json_done(s: str) -> bool:
    """True once `s` contains one complete top-level JSON object/array (stop decoding there)."""
    depth, in_str, esc = 0, False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return True
    return False


def first_json(s: str) -> str:
    """`s` cut right after its first complete top-level JSON value (drops trailing chatter)."""
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return s[:i + 1]
    return s


def scan_checkpoint(prefix: str) -> dict:
    """Where does a (possibly mid-token) extraction-JSON prefix stand? Returns the safe rollback
    text (ends after the last complete array item) and which top-level array it is in. Arrays
    appear in schema order: 1 = entities, 2 = triples."""
    depth, in_str, esc = 0, False, False
    arrays_opened, array_open, items, safe_end = 0, False, 0, 0
    for i, ch in enumerate(prefix):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if depth == 2 and array_open:      # a string item of a top-level array just closed
                    items, safe_end = items + 1, i + 1
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
            if ch == "[" and depth == 2:
                arrays_opened, array_open, items, safe_end = arrays_opened + 1, True, 0, i + 1
        elif ch in "}]":
            depth -= 1
            if ch == "}" and depth == 2 and array_open:  # an object item just closed
                items, safe_end = items + 1, i + 1
            elif ch == "]" and depth == 1:
                array_open, safe_end = False, i + 1
    return {"text": prefix[:safe_end], "arrays_opened": arrays_opened, "array_open": array_open, "items": items,
            "rolled_back_chars": len(prefix) - safe_end}


def closable(text: str) -> Optional[str]:
    """Online validator for a JSON stream: roll `text` back to its last complete array item and
    check that closing the document there parses. Returns that validated prefix ("" if too early
    to tell), or None on a syntax slip."""
    st = scan_checkpoint(text)
    if st["arrays_opened"] == 0:
        return "" if len(text) < 40 else None
    doc = st["text"] + ("]" if st["array_open"] else "") + "}"
    try:
        json.loads(doc)
    except json.JSONDecodeError:
        return None
    return st["text"]


# --------------------------------------------------------------------------- prompt rendering
class ChatML:
    """Raw-prompt renderer for ChatML templates (Qwen 2.5/3), verified against the live model."""

    @staticmethod
    def render(system: str, user: str, prefix: str = "", think_tag: bool = False) -> str:
        # Matches Ollama's qwen3 template with think=false token for token (prompt_eval_count).
        return (f"<|im_start|>system\n{system}\n\n<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{'<think>' + chr(10) if think_tag else ''}{prefix}")

    @staticmethod
    async def verify(client, model: str, system: str, think_tag: bool) -> bool:
        """True iff our raw render tokenizes exactly like Ollama's own chat render for `model`.
        Costs two 1-token requests at worker startup."""
        user = "Resume-capability probe: Pump P-1 feeds tank T-2."
        try:
            show = (await client.post("/api/show", json={"model": model}, timeout=10.0)).json()
            if "<|im_start|>" not in (show.get("template") or ""):
                return False
            # num_ctx must match the worker's inference requests: an unset value makes Ollama (re)load the
            # model at its default context, which for qwen3 is 262k tokens (~40 GB).
            opts = {"num_predict": 1, "num_ctx": 4096}
            chat = {"model": model, "stream": False, "options": opts,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if think_tag:
                chat["think"] = False
            a = (await client.post("/api/chat", json=chat, timeout=60.0)).json()
            b = (await client.post("/api/generate", timeout=60.0, json={
                "model": model, "raw": True, "stream": False, "options": opts,
                "prompt": ChatML.render(system, user, "", think_tag)})).json()
            return bool(a.get("prompt_eval_count")) and a.get("prompt_eval_count") == b.get("prompt_eval_count")
        except Exception:
            return False


# --------------------------------------------------------------------------- structured resume
ENTITY_ITEM = {"type": "string"}
TRIPLE_ITEM = {"type": "object", "properties": {"subject": {"type": "string"}, "relation": {"type": "string"},
                                                "object": {"type": "string"}},
               "required": ["subject", "relation", "object"]}

RawGen = Callable[[str, Optional[dict], Optional[int]], Awaitable[str]]


async def structured_resume(raw: RawGen, render: Callable[[str], str], prefix: str,
                            on_text: Optional[Callable[[str], None]] = None) -> str:
    """Continue a half-written extraction with every model-written token grammar-constrained.

    `raw(prompt, json_schema_or_None, num_predict_or_None)` runs one raw generation and returns its
    text (the caller counts tokens / honours cancellation); `render(prefix)` builds the raw prompt.
    `on_text(text)` is called whenever the document grows - it is always a valid checkpoint."""
    st = scan_checkpoint(prefix)
    if st["arrays_opened"] == 0:
        raise ValueError("checkpoint too short to resume")  # caller restarts from scratch (nothing lost)
    text = st["text"]

    def grow(s: str) -> None:
        nonlocal text
        text += s
        if on_text:
            on_text(text)

    async def fill_array(item_schema: dict, items: int) -> None:
        while items < MAX_ITEMS:
            # Peek from just BEFORE the last closer: tokenizers merge `},` / `",` into one token, so
            # "how does this item close" is in-distribution where "what follows `}`" is not (measured).
            peek = (await raw(render(text[:-1]), None, 4)).lstrip()
            after = peek[1:].lstrip() if peek[:1] == text[-1] else "?"
            more = after[:1] == "," if items else after[:1] in ('"', "{")
            if not more:
                break
            sep = ",\n    " if items else "\n    "
            item = await raw(render(text + sep), item_schema, None)
            grow(sep + item.strip())
            items += 1
        grow("\n  ]")

    which, is_open, items = st["arrays_opened"], st["array_open"], st["items"]
    if which == 1:
        if is_open:
            await fill_array(ENTITY_ITEM, items)
        grow(',\n  "triples": [')
        await fill_array(TRIPLE_ITEM, 0)
    elif is_open:
        await fill_array(TRIPLE_ITEM, items)
    grow("\n}")
    return text
