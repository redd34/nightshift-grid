"""GraphRAG extraction contract shared by workers (validation) and the coordinator (sink).

Small INT4 models are not perfectly reliable JSON emitters, so everything that crosses
the grid boundary goes through `normalize_extraction()`: it accepts the handful of shapes
3B-8B models actually produce and turns them into one canonical form:

    {"entities": [str, ...], "triples": [[subject, relation, object], ...]}
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol

SYSTEM_PROMPT = (
    "You are an edge GraphRAG extraction micro-worker. Extract the key entities and the "
    "relationships between them from the user's text. Respond with JSON only, using keys "
    "'entities' (list of entity names) and 'triples' (list of objects with 'subject', "
    "'relation', 'object'). Use short UPPER_SNAKE_CASE relation names."
)

# JSON schema passed to Ollama's `format` field (structured outputs): constrains decoding so
# small models cannot emit prose or truncated keys.
EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {"type": "string"},
                    "object": {"type": "string"},
                },
                "required": ["subject", "relation", "object"],
            },
        },
    },
    "required": ["entities", "triples"],
}

_TRIPLE_KEYS = (("subject", "relation", "object"), ("source", "relation", "target"), ("head", "type", "tail"))


class ExtractionError(ValueError):
    """Model output could not be coerced into the extraction contract."""


def _clean(value: Any) -> str | None:
    if isinstance(value, (str, int, float)) and str(value).strip():
        return str(value).strip()
    return None


def canonical_relation(rel: str) -> str:
    """'exceeded during' / 'exceededDuring' -> 'EXCEEDED_DURING' so edges merge across chunks/models."""
    rel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", rel)
    return re.sub(r"[^0-9A-Za-z]+", "_", rel).strip("_").upper()


def _coerce_triple(item: Any) -> list[str] | None:
    parts: list[str | None] | None = None
    if isinstance(item, (list, tuple)) and len(item) == 3:
        parts = [_clean(p) for p in item]
    elif isinstance(item, dict):
        for keys in _TRIPLE_KEYS:
            if all(k in item for k in keys):
                parts = [_clean(item[k]) for k in keys]
                break
    if not parts or not all(parts):
        return None
    rel = canonical_relation(parts[1])  # type: ignore[arg-type]
    return [parts[0], rel, parts[2]] if rel else None  # type: ignore[list-item]


def _find_span(candidate: str, source: str) -> str | None:
    i = source.casefold().find(candidate.casefold()) if candidate.strip() else -1
    return source[i:i + len(candidate)] if i >= 0 else None


def _alnum_key(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.casefold())


def repair_value(value: str, source: str) -> tuple[str, bool]:
    """Undo grammar-constrained-decoding corruption by re-grounding a value in the source text.

    Measured with qwen3:4b on Ollama 0.34: JSON-schema decoding injects stray ':' into values
    ('4.2 mm/s' -> '4.2 mm: s', 'B-120' -> 'B-12:0', 'B-117' -> 'B-1:117', '4.2 mm: mm/s'). 12
    corrupted values per 10 chunks with a schema, 6 with format=json, 0 unconstrained (but then
    30% of outputs aren't parseable). Extracted entities should be spans of the chunk, so a
    value containing a ':' that the source doesn't is repaired to the source span it came
    from. Only exact (case-insensitive) source spans are accepted, so this never invents text;
    values that can't be grounded are returned unchanged.
    """
    if ":" not in value or value.casefold() in source.casefold():
        return value, False
    for cand in (re.sub(r"(\w+):\s*(?=\1)", "", value),  # 'mm: mm/s' -> 'mm/s', '1:117' -> '117'
                 value.replace(": ", "/"),                 # 'mm: s' -> 'mm/s'
                 value.replace(":", ""),                   # 'B-12:0' -> 'B-120'
                 value.replace(": ", " "), value.replace(":", " ")):
        span = _find_span(cand, source)
        if span:
            return span, True
    key = _alnum_key(value)  # last resort: a 1-6 token source window with identical alphanumerics
    toks = list(re.finditer(r"\S+", source))
    for n in range(1, 7):
        for i in range(len(toks) - n + 1):
            span = source[toks[i].start():toks[i + n - 1].end()].strip(".,;")
            if _alnum_key(span) == key:
                return span, True
    return value, False


def normalize_extraction(raw: str | dict, source_text: str | None = None) -> dict:
    """Coerce model output into canonical form. Raises ExtractionError if unusable.

    With `source_text`, entity names and triple endpoints are re-grounded in the chunk (see
    `repair_value`); `repaired_values` counts how many were fixed."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ExtractionError(f"model output is not JSON: {e.msg}") from e
    if not isinstance(raw, dict):
        raise ExtractionError(f"expected JSON object, got {type(raw).__name__}")

    repaired = 0

    def ground(v: str) -> str:
        nonlocal repaired
        if source_text is None:
            return v
        fixed, changed = repair_value(v, source_text)
        repaired += changed
        return fixed

    entities: list[str] = []
    for e in raw.get("entities") or []:
        name = _clean(e.get("name") if isinstance(e, dict) else e)
        name = ground(name) if name else name
        if name and name not in entities:
            entities.append(name)

    triples: list[list[str]] = []
    dropped = 0
    for t in raw.get("triples") or raw.get("relationships") or []:
        coerced = _coerce_triple(t)
        if coerced is not None:
            coerced = [ground(coerced[0]), coerced[1], ground(coerced[2])]
        if coerced is None:
            dropped += 1
        elif coerced not in triples:
            triples.append(coerced)

    # Entities mentioned only inside triples are still entities of the graph.
    for s, _, o in triples:
        for name in (s, o):
            if name not in entities:
                entities.append(name)

    if not entities and not triples:
        raise ExtractionError("extraction contained no entities or triples")
    return {"entities": entities, "triples": triples, "dropped_triples": dropped, "repaired_values": repaired}


class GraphSink(Protocol):
    def write(self, record: dict) -> None: ...
    def close(self) -> None: ...


class NullSink:
    def write(self, record: dict) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlGraphSink:
    """Append-only JSON Lines sink, idempotent per `task_id`.

    Each record is flushed and fsynced. The task_id is marked written as soon as the bytes are
    handed to the OS. So if fsync fails and the coordinator retries the completion, the
    retry is a no-op instead of a duplicate line. Existing task_ids are loaded on open, so
    dedupe survives a coordinator restart that appends to the same file."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._written: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self._written.add(json.loads(line)["task_id"])
                except (ValueError, KeyError, TypeError):
                    pass  # tolerate a torn final line
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if record["task_id"] in self._written:
                return
            self._fh.write(line + "\n")
            self._fh.flush()
            self._written.add(record["task_id"])
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()
