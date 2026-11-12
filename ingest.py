"""Map-side input for the Nightshift grid: turn files on disk into GraphRAG extraction chunks.

    python ingest.py docs/ logs/plant.log --tier 2 --job-name "Plant logs" --wait
    python ingest.py samples/ --dry-run

Pipeline: walk paths -> extract text per file type -> split into paragraph/sentence units ->
greedily pack units into ~`--chunk-tokens` windows with `--overlap` tokens of trailing
context -> dedupe identical chunks by sha256 -> submit to the coordinator in jobs of at most
`MAX_CHUNKS_PER_JOB` chunks.

Token budget. We do not ship a tokenizer to every edge box just to size chunks, so tokens are
estimated from words (split on whitespace and , ; |): ``ceil(words * 4 / 3)`` (BPE tokenizers average
~0.75 English words per token). That is within ~15% for prose and log lines, which is plenty
for a budget whose only purpose is keeping each prompt well inside a 3B model's context.

Provenance: sidecar manifest, NOT an in-band header. The coordinator payload is a plain
string that goes straight into the extraction prompt; a ``[source: path#chunk-3]`` header
would be extracted as an entity by small models often enough to pollute the graph, would
change the dedupe hash for identical text from different files, and leaks file paths to
every worker. Instead we write ``<manifest-dir>/<job_id>.manifest.jsonl`` with one line per
chunk: ``{job_id, chunk_index, path, start, end, sha256, approx_tokens, also_in}``. The
coordinator already stamps (job_id, chunk_index) on every output record, so the reduce step
joins on that pair (``reduce_graph.py --manifest``) to cite file + character offsets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

MAX_CHUNKS_PER_JOB = 5000
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".csv", ".tsv", ".rst"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | JSON_SUFFIXES | PDF_SUFFIXES
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}
# JSON keys that usually hold the human-readable body of a record (checked first).
PREFERRED_JSON_KEYS = ("text", "content", "body", "message", "description", "summary", "notes", "title")
MIN_JSON_STRING_WORDS = 4  # fallback: any string field with at least this many words


def approx_tokens(text: str) -> int:
    """Word-based token estimate (see module docstring): ceil(words * 4/3)."""
    # Commas/semicolons/pipes count as word breaks so dense CSV rows aren't badly undercounted.
    words = len(re.findall(r"[^\s,;|]+", text))
    return math.ceil(words * 4 / 3)


def _tokens_to_words(tokens: int) -> int:
    return max(1, (tokens * 3) // 4)


# --------------------------------------------------------------------------- file discovery


def iter_files(paths: Iterable[str | os.PathLike]) -> Iterator[Path]:
    """Yield supported files under `paths` in a stable (sorted) order, skipping tool/VCS dirs."""
    seen: set[Path] = set()
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            candidates: Iterable[Path] = [p]
        elif p.is_dir():
            candidates = sorted(
                f for f in p.rglob("*")
                if f.is_file() and not any(part in SKIP_DIRS or part.startswith(".") for part in f.relative_to(p).parts)
            )
        else:
            print(f"warning: {p} does not exist, skipping", file=sys.stderr)
            continue
        for f in candidates:
            if f.suffix.lower() in SUPPORTED_SUFFIXES and f.resolve() not in seen:
                seen.add(f.resolve())
                yield f


def looks_binary(data: bytes) -> bool:
    """NUL bytes in the first 8KB are a reliable signal of a binary file mislabeled as text."""
    return b"\x00" in data[:8192]


def _json_strings(obj: object) -> list[str]:
    """Pull prose out of a JSON value: preferred keys first, else any long-ish string field."""
    if isinstance(obj, dict):
        preferred = [obj[k] for k in PREFERRED_JSON_KEYS if isinstance(obj.get(k), str) and obj[k].strip()]
        if preferred:
            return [s.strip() for s in preferred]
        out: list[str] = []
        for v in obj.values():
            out.extend(_json_strings(v))
        return out
    if isinstance(obj, list):
        out = []
        for v in obj:
            out.extend(_json_strings(v))
        return out
    if isinstance(obj, str) and len(obj.split()) >= MIN_JSON_STRING_WORDS:
        return [obj.strip()]
    return []


def extract_json_text(raw: str, jsonl: bool) -> str:
    """Text fields of a .json document or each .jsonl line, one record per paragraph."""
    parts: list[str] = []
    if jsonl:  # identical records (repeated alerts/log lines) are kept once
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parts.extend(_json_strings(json.loads(line)))
            except json.JSONDecodeError:
                continue  # tolerate a torn line
    else:
        parts.extend(_json_strings(json.loads(raw)))
    return "\n\n".join(dict.fromkeys(parts))


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # optional dependency
    except ImportError as e:  # pragma: no cover - depends on environment
        raise RuntimeError(f"{path}: PDF support needs `pypdf` (uv pip install pypdf)") from e
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)


def read_document(path: Path) -> str | None:
    """Return the extractable text of `path`, or None if it is empty/binary/unparseable."""
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        text = read_pdf_text(path)
    else:
        data = path.read_bytes()
        if not data.strip() or looks_binary(data):
            return None
        raw = data.decode("utf-8", errors="replace")
        if suffix in JSON_SUFFIXES:
            try:
                text = extract_json_text(raw, jsonl=suffix != ".json")
            except json.JSONDecodeError:
                print(f"warning: {path} is not valid JSON, skipping", file=sys.stderr)
                return None
        else:
            text = raw
    return text if text and text.strip() else None


# --------------------------------------------------------------------------- chunking

_PARA_RE = re.compile(r"\S(?:.*?\S)?(?=[ \t]*\n\s*\n|\s*\Z)", re.S)
# Sentence end: . ! ? followed by whitespace, but not inside decimals like "4.2 mm/s".
_SENT_RE = re.compile(r"\S.*?(?:[.!?](?=\s)|\Z)", re.S)


@dataclass(frozen=True)
class Unit:
    """A paragraph, sentence or word window with its character span in the source text."""
    start: int
    end: int
    tokens: int


@dataclass
class Chunk:
    text: str
    path: str
    start: int
    end: int
    approx_tokens: int
    sha256: str = ""
    also_in: list[str] = field(default_factory=list)  # other locations with identical text


def _word_windows(text: str, start: int, end: int, max_tokens: int) -> list[Unit]:
    """Last resort for a single run-on sentence/line longer than the budget."""
    words = list(re.finditer(r"\S+", text[start:end]))
    step = _tokens_to_words(max_tokens)
    units = []
    for i in range(0, len(words), step):
        group = words[i:i + step]
        s, e = start + group[0].start(), start + group[-1].end()
        units.append(Unit(s, e, approx_tokens(text[s:e])))
    return units


def split_units(text: str, max_tokens: int) -> list[Unit]:
    """Paragraphs if they fit the budget, else sentences, else word windows.

    Splitting on the largest natural boundary that fits keeps related facts (e.g. one
    maintenance log entry) in the same chunk, which is what entity/relation extraction needs.
    """
    units: list[Unit] = []
    for para in _PARA_RE.finditer(text):
        ptoks = approx_tokens(para.group())
        if ptoks <= max_tokens:
            units.append(Unit(para.start(), para.end(), ptoks))
            continue
        for sent in _SENT_RE.finditer(text, para.start(), para.end()):
            s, e = sent.start(), sent.end()
            stoks = approx_tokens(text[s:e])
            if stoks <= max_tokens:
                units.append(Unit(s, e, stoks))
            else:
                units.extend(_word_windows(text, s, e, max_tokens))
    return units


def chunk_text(text: str, chunk_tokens: int = 350, overlap: int = 50) -> list[tuple[int, int]]:
    """Greedily pack units into spans of <= chunk_tokens; each new span re-includes trailing
    units of the previous one totalling <= overlap tokens, so a relation split across a
    boundary is still seen whole by at least one worker. Returns (start, end) char spans."""
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    overlap = max(0, min(overlap, chunk_tokens // 2))
    units = split_units(text, chunk_tokens)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (j == i or total + units[j].tokens <= chunk_tokens):
            total += units[j].tokens
            j += 1
        spans.append((units[i].start, units[j - 1].end))
        if j >= len(units):
            break
        # Step back over trailing units that fit the overlap budget (but always make progress).
        k, back = j, 0
        while k - 1 > i and back + units[k - 1].tokens <= overlap:
            back += units[k - 1].tokens
            k -= 1
        i = k
    return spans


def _normalize_for_hash(text: str) -> str:
    return " ".join(text.split())


def chunk_files(paths: Iterable[str | os.PathLike], chunk_tokens: int = 350, overlap: int = 50) -> tuple[list[Chunk], dict]:
    """Chunk every supported file and dedupe identical chunk text (whitespace-insensitive).

    Duplicates are common in real corpora (boilerplate, re-exported reports, repeated log
    lines) and every duplicate costs a full model call on some edge node, so we keep the
    first occurrence and record the others in `also_in` for provenance."""
    chunks: list[Chunk] = []
    by_hash: dict[str, Chunk] = {}
    stats = {"files": 0, "skipped_files": [], "duplicate_chunks": 0}
    for path in iter_files(paths):
        try:
            text = read_document(path)
        except (RuntimeError, OSError, ValueError) as e:
            stats["skipped_files"].append(f"{path}: {e}")
            continue
        if text is None:
            stats["skipped_files"].append(f"{path}: empty or binary")
            continue
        stats["files"] += 1
        for s, e in chunk_text(text, chunk_tokens, overlap):
            body = text[s:e]
            digest = hashlib.sha256(_normalize_for_hash(body).encode("utf-8")).hexdigest()
            if digest in by_hash:
                by_hash[digest].also_in.append(f"{path}:{s}-{e}")
                stats["duplicate_chunks"] += 1
                continue
            c = Chunk(text=body, path=str(path), start=s, end=e, approx_tokens=approx_tokens(body), sha256=digest)
            by_hash[digest] = c
            chunks.append(c)
    return chunks, stats


def batch(items: list, size: int = MAX_CHUNKS_PER_JOB) -> list[list]:
    """Split into jobs of <= size so one huge corpus doesn't become one unbounded job."""
    return [items[i:i + size] for i in range(0, len(items), size)] or []


def manifest_lines(job_id: str, chunks: list[Chunk]) -> list[dict]:
    """chunk_index is the position in the submitted list (the coordinator assigns it that way)."""
    return [
        {"job_id": job_id, "chunk_index": idx, "path": c.path, "start": c.start, "end": c.end,
         "sha256": c.sha256, "approx_tokens": c.approx_tokens, "also_in": c.also_in}
        for idx, c in enumerate(chunks)
    ]


def write_manifest(manifest_dir: Path, job_id: str, chunks: list[Chunk]) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{job_id}.manifest.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for line in manifest_lines(job_id, chunks):
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--coordinator", default=os.environ.get("GRID_COORDINATOR", "http://127.0.0.1:8000"))
    ap.add_argument("--chunk-tokens", type=int, default=350)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--tier", type=int, default=2, help="required capability tier (2 = JSON extraction)")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--manifest-dir", default="manifests", help="where <job_id>.manifest.jsonl is written")
    ap.add_argument("--dry-run", action="store_true", help="chunk and report, submit nothing, write nothing")
    ap.add_argument("--wait", action="store_true", help="block until every submitted job finishes")
    a = ap.parse_args(argv)

    chunks, stats = chunk_files(a.paths, a.chunk_tokens, a.overlap)
    for s in stats["skipped_files"]:
        print(f"skipped {s}", file=sys.stderr)
    toks = [c.approx_tokens for c in chunks]
    print(f"{stats['files']} files -> {len(chunks)} chunks ({stats['duplicate_chunks']} duplicates dropped); "
          f"approx tokens/chunk min={min(toks, default=0)} max={max(toks, default=0)} "
          f"mean={sum(toks) / max(1, len(toks)):.0f}")
    if not chunks:
        return 1
    jobs = batch(chunks)
    name = a.job_name or f"Ingest {', '.join(Path(p).name for p in a.paths)[:60]}"

    if a.dry_run:
        for c in chunks[:5]:
            preview = " ".join(c.text.split())[:100]
            print(f"  {c.path}[{c.start}:{c.end}] ~{c.approx_tokens}t  {preview}...")
        print(f"dry run: would submit {len(jobs)} job(s) at tier {a.tier} to {a.coordinator}")
        return 0

    import demo_submit  # imported lazily so --dry-run works without httpx/coordinator

    submitted = []
    for n, job_chunks in enumerate(jobs, 1):
        job_name = name if len(jobs) == 1 else f"{name} (part {n}/{len(jobs)})"
        job = demo_submit.submit(a.coordinator, chunks=[c.text for c in job_chunks],
                                 required_tier=a.tier, job_name=job_name)
        path = write_manifest(Path(a.manifest_dir), job["job_id"], job_chunks)
        print(f"submitted job {job['job_id']} ({len(job_chunks)} chunks) manifest={path}")
        submitted.append(job["job_id"])
    if a.wait:
        for job_id in submitted:
            st = demo_submit.wait(a.coordinator, job_id)
            print(f"job {job_id} done in {st.get('elapsed_s', 0):.1f}s: {st.get('counts')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
