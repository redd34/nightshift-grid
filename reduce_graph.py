"""Reduce side of Nightshift GraphRAG: per-chunk extractions -> one resolved knowledge graph.

    python reduce_graph.py output_graph.jsonl --out graph.json [--embed-model nomic-embed-text]
                           [--qdrant http://localhost:6333|PATH|:memory:|none] [--manifest m.jsonl]
    python reduce_graph.py query graph.json "which bearing assemblies were replaced?" [--hops 1]

Each map task emits {entities, triples} for one chunk, written by many small models, so the
same real-world thing shows up as "PLC-Node-2", "plc node 2", "PLC Node 2", "Compressor:
turbine"... The reduce step:

1. **Canonical keys** (`normalize_key`): NFKC + casefold, separators/punctuation -> space,
   letter/digit boundaries split ("node2" == "node 2"), decimals kept ("4.2"), leading
   articles stripped. Identical keys merge unconditionally.
2. **Optional embedding merge** of remaining near-duplicates (cosine >= --similarity) -- but
   only among *candidate pairs* that share a blocking key (a content token or a 4-char
   prefix), with oversized blocks skipped, so cost is ~sum(block^2) instead of O(n^2).
3. **Merge guards** (`can_merge`): differing digit sequences never merge ("B-101" vs "B-102",
   "4.2 mm/s" vs "4.5 mm/s"), even at cosine 0.99 -- embeddings are notoriously blind to
   numbers, and wrongly fusing two assets is far worse for maintenance RAG than a duplicate.
4. **Union-find clusters**; canonical name = most frequent surface form; others kept as aliases.
5. **Graph**: deduped edges (source, RELATION, target) with weight and chunk-level evidence.
6. **Sinks**: graph.json (+ GraphML) and optionally Qdrant (`nightshift_*` collections only).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from relay import DEFAULT_NUM_CTX
from pathlib import Path
from typing import Callable, Iterable, Sequence
from xml.etree import ElementTree as ET

DEFAULT_OLLAMA = "http://127.0.0.1:11434"
REQUIRED_COLLECTION_PREFIX = "nightshift_"
NAMESPACE = uuid.UUID("6f1c3a52-6b7e-4d2b-9a43-5e0c1d0b7a11")  # uuid5 namespace for point ids
ARTICLES = {"the", "a", "an"}
STOPWORDS = ARTICLES | {
    "of", "on", "in", "at", "to", "for", "and", "or", "by", "with", "from", "is", "was", "were",
    "be", "been", "are", "which", "what", "who", "whom", "when", "where", "why", "how", "did",
    "do", "does", "that", "this", "these", "those", "it", "its", "any", "all", "has", "have", "had",
}
MAX_BLOCK = 50  # a blocking key shared by more names than this is too generic to be useful

Embedder = Callable[[Sequence[str]], list[list[float]]]


# --------------------------------------------------------------------------- loading


def load_records(path: str | Path, job: str | None = None) -> tuple[list[dict], dict]:
    """Read the coordinator's JSONL sink. Tolerates torn/garbage lines (a crash mid-append) and
    keeps the first record per task_id (the sink is idempotent, but files get concatenated)."""
    records: list[dict] = []
    seen: set[str] = set()
    stats = {"lines": 0, "bad_lines": 0, "duplicates": 0, "filtered": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict) or "task_id" not in rec:
                    raise ValueError("not a record")
            except ValueError:
                stats["bad_lines"] += 1
                continue
            if job and rec.get("job_id") != job:
                stats["filtered"] += 1
                continue
            if rec["task_id"] in seen:
                stats["duplicates"] += 1
                continue
            seen.add(rec["task_id"])
            records.append(rec)
    return records, stats


def load_manifests(paths: Iterable[str | Path]) -> dict[tuple[str, int], dict]:
    """(job_id, chunk_index) -> {path, start, end, sha256} from ingest.py sidecar manifests."""
    out: dict[tuple[str, int], dict] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            try:
                m = json.loads(line)
                out[(m["job_id"], int(m["chunk_index"]))] = {k: m[k] for k in ("path", "start", "end", "sha256") if k in m}
            except (ValueError, KeyError, TypeError):
                continue
    return out


# --------------------------------------------------------------------------- normalization


_SEP_RE = re.compile(r"(?<!\d)[.,]|[.,](?!\d)|[^\w.,]|_")  # punctuation/separators, but keep 4.2 / 1,000
_DIGITS_RE = re.compile(r"\d+(?:[.,]\d+)*")


def normalize_key(name: str) -> str:
    """Canonical comparison key for an entity surface form.

    'PLC-Node-2' / 'plc node 2' / 'PLC Node 2' / 'PLC_Node2' -> 'plc node 2'
    'Compressor: turbine' -> 'compressor turbine';  'The Technician' -> 'technician'
    Decimal points between digits survive so '4.2 mm/s' != '42 mm/s'."""
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", s)  # letter<->digit boundary
    s = _SEP_RE.sub(" ", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 1,000 -> 1000
    tokens = s.split()
    while len(tokens) > 1 and tokens[0] in ARTICLES:
        tokens.pop(0)
    return " ".join(tokens)


def digit_signature(key: str) -> tuple[str, ...]:
    """The ordered numeric tokens of a key. Two names may only merge if these are identical."""
    return tuple(_DIGITS_RE.findall(key))


def can_merge(key_a: str, key_b: str) -> bool:
    """Hard guard applied to every non-identical merge (embedding or otherwise).

    Identifiers and measurements differ only in their digits ('B-101'/'B-102',
    '4.2 mm/s'/'4.5 mm/s'), which is exactly where embeddings are weakest. Also refuses to
    merge a name with digits into one without ('pump 3' vs 'pump')."""
    return bool(key_a) and bool(key_b) and digit_signature(key_a) == digit_signature(key_b)


def _stem(token: str) -> str:
    """Tiny plural stemmer for blocking/keyword matching only (never for keys)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [_stem(t) for t in normalize_key(text).split() if t not in STOPWORDS and not t[0].isdigit() and len(t) > 1]


def blocking_keys(key: str) -> set[str]:
    """Candidate-generation keys: stemmed content tokens (>=3 chars) plus a 4-char prefix.

    Two names are only ever compared if they share one of these, which keeps embedding
    merge near-linear on large graphs while still catching 'turbine vibration' ~ 'vibrations'."""
    keys = {f"t:{t}" for t in content_tokens(key) if len(t) >= 3}
    compact = key.replace(" ", "")
    if len(compact) >= 4:
        keys.add(f"p:{compact[:4]}")
    return keys


def candidate_pairs(keys: Sequence[str], max_block: int = MAX_BLOCK) -> set[tuple[str, str]]:
    """All (a, b) with a < b sharing a blocking key, from blocks no larger than max_block,
    pre-filtered by `can_merge` so we never embed/compare pairs that could not merge anyway."""
    blocks: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        for b in blocking_keys(k):
            blocks[b].append(k)
    pairs: set[tuple[str, str]] = set()
    for members in blocks.values():
        if len(members) < 2 or len(members) > max_block:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                if a != b and can_merge(a, b):
                    pairs.add((a, b) if a < b else (b, a))
    return pairs


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v))
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


class UnionFind:
    def __init__(self, items: Iterable[str] = ()):
        self.parent: dict[str, str] = {}
        for i in items:
            self.add(i)

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller key becomes root
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


# --------------------------------------------------------------------------- resolution


@dataclass
class Resolution:
    """Mapping from every surface form to a cluster, plus cluster metadata."""
    surface_to_node: dict[str, str] = field(default_factory=dict)
    nodes: dict[str, dict] = field(default_factory=dict)  # node_id -> {name, aliases, keys}
    raw_surface_forms: int = 0
    key_clusters: int = 0
    embedding_merges: list[tuple[str, str, float]] = field(default_factory=list)


def _node_id(key: str) -> str:
    return re.sub(r"\s+", "_", key) or "_"


def _surface_counts(records: Iterable[dict]) -> Counter:
    """Mentions per surface form: once per chunk for each entity listed or used in a triple."""
    counts: Counter = Counter()
    for r in records:
        names = set(r.get("entities") or [])
        for t in r.get("triples") or []:
            if len(t) == 3:
                names.update((t[0], t[2]))
        counts.update(n for n in names if isinstance(n, str) and n.strip())
    return counts


def resolve_entities(records: Sequence[dict], embedder: Embedder | None = None,
                     similarity: float = 0.9, max_block: int = MAX_BLOCK) -> Resolution:
    """Cluster entity surface forms: exact canonical-key merge, then (optionally) guarded
    embedding merge over blocked candidate pairs. Pure apart from calling `embedder`."""
    counts = _surface_counts(records)
    res = Resolution(raw_surface_forms=len(counts))
    key_of = {s: normalize_key(s) for s in counts}
    keys = sorted({k for k in key_of.values() if k})
    res.key_clusters = len(keys)
    uf = UnionFind(keys)

    if embedder is not None and len(keys) > 1:
        pairs = sorted(candidate_pairs(keys, max_block))
        needed = sorted({k for p in pairs for k in p})
        if needed:
            # Embed the most frequent surface form for each key: it reads more naturally than the key.
            best_surface = {}
            for s, c in counts.most_common():
                best_surface.setdefault(key_of[s], s)
            vectors = dict(zip(needed, embedder([best_surface[k] for k in needed])))
            for a, b in pairs:
                sim = cosine(vectors[a], vectors[b])
                if sim >= similarity and can_merge(a, b):
                    uf.union(a, b)
                    res.embedding_merges.append((a, b, round(sim, 4)))

    clusters: dict[str, list[str]] = defaultdict(list)
    for s in counts:
        if key_of[s]:
            clusters[uf.find(key_of[s])].append(s)
    for root, surfaces in clusters.items():
        # Most frequent surface form wins; ties -> longer (more specific), then lexical.
        surfaces.sort(key=lambda s: (-counts[s], -len(s), s))
        canonical = surfaces[0]
        node_id = _node_id(normalize_key(canonical))
        res.nodes[node_id] = {"name": canonical, "aliases": surfaces[1:],
                              "keys": sorted({key_of[s] for s in surfaces})}
        for s in surfaces:
            res.surface_to_node[s] = node_id
    return res


# --------------------------------------------------------------------------- graph


def _chunk_ref(rec: dict) -> str:
    return f"{rec.get('job_id')}#{rec.get('chunk_index')}"


def build_graph(records: Sequence[dict], res: Resolution,
                manifest: dict[tuple[str, int], dict] | None = None) -> dict:
    """Assemble nodes/edges after resolution. Edges are deduped on (source, RELATION, target)
    with weight = number of distinct chunks asserting them; self-loops created by merging
    (e.g. 'Compressor turbine' EXCEEDED 'compressor turbine') are dropped and counted."""
    manifest = manifest or {}
    mentions: Counter = Counter()
    node_chunks: dict[str, set[str]] = defaultdict(set)
    edges: dict[tuple[str, str, str], dict] = {}
    self_loops = 0
    raw_triples = 0
    chunk_meta: dict[str, dict] = {}

    for r in records:
        ref = _chunk_ref(r)
        meta = {"job_id": r.get("job_id"), "chunk_index": r.get("chunk_index"), "task_id": r.get("task_id")}
        src = manifest.get((r.get("job_id"), r.get("chunk_index")))
        if src:
            meta["source"] = src
        chunk_meta[ref] = meta
        names = set(r.get("entities") or [])
        for t in r.get("triples") or []:
            if len(t) == 3:
                names.update((t[0], t[2]))
        for n in names:
            nid = res.surface_to_node.get(n)
            if nid:
                mentions[nid] += 1
                node_chunks[nid].add(ref)
        seen_here: set[tuple[str, str, str]] = set()
        for t in r.get("triples") or []:
            if len(t) != 3:
                continue
            raw_triples += 1
            s, rel, o = res.surface_to_node.get(t[0]), str(t[1]), res.surface_to_node.get(t[2])
            if not s or not o or not rel:
                continue
            if s == o:
                self_loops += 1
                continue
            k = (s, rel, o)
            e = edges.setdefault(k, {"source": s, "relation": rel, "target": o, "weight": 0, "evidence": []})
            if k not in seen_here:  # one chunk asserting the same fact twice counts once
                seen_here.add(k)
                e["weight"] += 1
                e["evidence"].append(dict(meta))

    nodes = []
    for nid, info in sorted(res.nodes.items()):
        nodes.append({"id": nid, "name": info["name"], "aliases": info["aliases"],
                      "mention_count": mentions[nid], "chunks": sorted(node_chunks[nid], key=_ref_sort)})
    edge_list = sorted(edges.values(), key=lambda e: (-e["weight"], e["source"], e["relation"], e["target"]))
    return {
        "nodes": nodes,
        "edges": edge_list,
        "chunks": {ref: chunk_meta[ref] for ref in sorted(chunk_meta, key=_ref_sort)},
        "stats": {"records": len(records), "raw_surface_forms": res.raw_surface_forms,
                  "key_clusters": res.key_clusters, "nodes": len(nodes), "raw_triples": raw_triples,
                  "edges": len(edge_list), "self_loops_dropped": self_loops,
                  "embedding_merges": len(res.embedding_merges)},
    }


def _ref_sort(ref: str) -> tuple[str, int]:
    job, _, idx = ref.partition("#")
    return (job, int(idx) if idx.lstrip("-").isdigit() else -1)


def degrees(graph: dict) -> Counter:
    d: Counter = Counter()
    for e in graph["edges"]:
        d[e["source"]] += 1
        d[e["target"]] += 1
    return d


def write_graphml(graph: dict, path: str | Path) -> None:
    """GraphML for Gephi / Neo4j (apoc.import.graphml) / networkx, using only the stdlib."""
    root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    for kid, target, name, typ in [("name", "node", "name", "string"), ("aliases", "node", "aliases", "string"),
                                   ("mentions", "node", "mention_count", "int"),
                                   ("relation", "edge", "relation", "string"), ("weight", "edge", "weight", "int"),
                                   ("evidence", "edge", "evidence", "string")]:
        ET.SubElement(root, "key", id=kid, attrib={"for": target, "attr.name": name, "attr.type": typ})
    g = ET.SubElement(root, "graph", id="nightshift", edgedefault="directed")

    def data(parent: ET.Element, key: str, value: object) -> None:
        ET.SubElement(parent, "data", key=key).text = str(value)

    for n in graph["nodes"]:
        el = ET.SubElement(g, "node", id=n["id"])
        data(el, "name", n["name"])
        data(el, "aliases", " | ".join(n["aliases"]))
        data(el, "mentions", n["mention_count"])
    for i, e in enumerate(graph["edges"]):
        el = ET.SubElement(g, "edge", id=f"e{i}", source=e["source"], target=e["target"])
        data(el, "relation", e["relation"])
        data(el, "weight", e["weight"])
        data(el, "evidence", ",".join(f"{ev['job_id']}#{ev['chunk_index']}" for ev in e["evidence"]))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def summarize(graph: dict, load_stats: dict | None = None, top: int = 8) -> str:
    st = graph["stats"]
    raw, n = st["raw_surface_forms"], st["nodes"]
    names = {x["id"]: x["name"] for x in graph["nodes"]}
    lines = []
    if load_stats:
        lines.append(f"lines={load_stats['lines']} bad={load_stats['bad_lines']} dup_task_ids={load_stats['duplicates']} "
                     f"filtered_by_job={load_stats['filtered']}")
    lines.append(f"records={st['records']}  raw entities={raw} -> key clusters={st['key_clusters']} -> nodes={n} "
                 f"(merge ratio {raw / max(1, n):.2f}x, {st['embedding_merges']} embedding merges)")
    lines.append(f"raw triples={st['raw_triples']} -> edges={st['edges']} (self-loops dropped={st['self_loops_dropped']})")
    lines.append("top entities by degree: " + ", ".join(f"{names[k]}({v})" for k, v in degrees(graph).most_common(top)))
    rels: Counter = Counter()
    for e in graph["edges"]:
        rels[e["relation"]] += e["weight"]
    lines.append("top relations: " + ", ".join(f"{k}({v})" for k, v in rels.most_common(top)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- embeddings / Ollama


class OllamaEmbedder:
    """Batched /api/embed client. nomic-embed-text expects task prefixes; `prefix` adds one."""

    def __init__(self, model: str = "nomic-embed-text", url: str = DEFAULT_OLLAMA, prefix: str = "", batch: int = 64):
        self.model, self.url, self.prefix, self.batch = model, url.rstrip("/"), prefix, batch

    def with_prefix(self, prefix: str) -> "OllamaEmbedder":
        return OllamaEmbedder(self.model, self.url, prefix, self.batch)

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = [self.prefix + t for t in texts[i:i + self.batch]]
            r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": part}, timeout=120)
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out


def _nomic_prefix(model: str, task: str) -> str:
    return f"{task}: " if "nomic" in model else ""


def make_embedder(model: str | None, url: str) -> OllamaEmbedder | None:
    """Return an embedder if the model is usable, else None with a clear message (never raise)."""
    if not model or model.lower() == "none":
        return None
    try:
        emb = OllamaEmbedder(model, url)
        emb(["probe"])
        return emb
    except Exception as e:  # network, 404 model not pulled, ...
        print(f"note: embedding model {model!r} unavailable at {url} ({e}); "
              f"continuing with key-based resolution only (try `ollama pull {model}`)", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- Qdrant sink


def collection_names(prefix: str) -> dict[str, str]:
    """Validate the prefix and derive collection names. The prefix MUST start with
    'nightshift_': the Qdrant server is shared with other projects and this tool must never be
    pointed at their collections, even by a typo on the command line."""
    if not prefix.startswith(REQUIRED_COLLECTION_PREFIX):
        raise ValueError(f"Qdrant collection prefix must start with {REQUIRED_COLLECTION_PREFIX!r}, got {prefix!r}")
    names = {"entities": f"{prefix}entities", "chunks": f"{prefix}chunks"}
    for n in names.values():
        _check_collection_name(n)
    return names


def _check_collection_name(name: str) -> None:
    if not name.startswith(REQUIRED_COLLECTION_PREFIX):
        raise ValueError(f"refusing to touch Qdrant collection {name!r}: only {REQUIRED_COLLECTION_PREFIX}* is allowed")


def open_qdrant(location: str):
    """URL -> remote server, ':memory:' -> in-process, anything else -> embedded local path."""
    from qdrant_client import QdrantClient  # optional dependency

    if location.startswith(("http://", "https://")):
        # We only use long-stable APIs (collection_exists/create/upsert/query_points).
        return QdrantClient(url=location, check_compatibility=False)
    if location == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=location)


class QdrantSink:
    """Idempotent upserts into `<prefix>entities` / `<prefix>chunks`.

    Safety contract (shared server): only nightshift_* names, never delete/recreate a
    collection, and an existing collection with a different vector size/distance is an error,
    not something to "fix" by recreating it. Point ids are uuid5(entity key / task_id), so
    re-running reduce overwrites the same points instead of duplicating them."""

    def __init__(self, client, prefix: str = REQUIRED_COLLECTION_PREFIX):
        self.client = client
        self.names = collection_names(prefix)

    def ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client import models

        _check_collection_name(name)
        if self.client.collection_exists(name):
            vectors = self.client.get_collection(name).config.params.vectors
            if not isinstance(vectors, models.VectorParams):
                raise RuntimeError(f"Qdrant collection {name!r} uses named/multi vectors; expected one unnamed "
                                   f"{dim}-d cosine vector. Use a different --qdrant-collection-prefix.")
            if vectors.size != dim or vectors.distance != models.Distance.COSINE:
                raise RuntimeError(f"Qdrant collection {name!r} exists with size={vectors.size} distance={vectors.distance}, "
                                   f"but this run needs size={dim} distance=Cosine. Refusing to recreate it; "
                                   f"use a different --qdrant-collection-prefix or a matching --embed-model.")
            return
        self.client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def _upsert(self, name: str, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> int:
        from qdrant_client import models

        _check_collection_name(name)
        if not ids:
            return 0
        self.ensure_collection(name, len(vectors[0]))
        for i in range(0, len(ids), 256):
            self.client.upsert(name, points=[models.PointStruct(id=pid, vector=v, payload=p)
                                             for pid, v, p in zip(ids[i:i + 256], vectors[i:i + 256], payloads[i:i + 256])])
        return len(ids)

    def upsert_entities(self, graph: dict, embedder: Embedder) -> int:
        nodes = graph["nodes"]
        texts = [entity_text(n) for n in nodes]
        return self._upsert(self.names["entities"], [entity_point_id(n["id"]) for n in nodes], embedder(texts) if texts else [],
                            [{"node_id": n["id"], "name": n["name"], "aliases": n["aliases"],
                              "mention_count": n["mention_count"], "chunks": n["chunks"]} for n in nodes])

    def upsert_chunks(self, records: Sequence[dict], res: Resolution, embedder: Embedder) -> int:
        recs = [r for r in records if r.get("source_text")]
        return self._upsert(self.names["chunks"], [chunk_point_id(r["task_id"]) for r in recs],
                            embedder([r["source_text"] for r in recs]) if recs else [],
                            [{"job_id": r.get("job_id"), "chunk_index": r.get("chunk_index"), "task_id": r["task_id"],
                              "entities": sorted({res.surface_to_node.get(e, e) for e in r.get("entities") or []}),
                              "triples": r.get("triples") or [], "text": r["source_text"]} for r in recs])

    def search_entities(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        name = self.names["entities"]
        if not self.client.collection_exists(name):
            return []
        hits = self.client.query_points(name, query=vector, limit=k, with_payload=True).points
        return [(h.payload["node_id"], h.score) for h in hits]


def entity_text(node: dict) -> str:
    aliases = [a for a in node["aliases"] if normalize_key(a) != normalize_key(node["name"])][:5]
    return node["name"] + (f" (also: {', '.join(aliases)})" if aliases else "")


def entity_point_id(node_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"entity:{node_id}"))


def chunk_point_id(task_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"chunk:{task_id}"))


# --------------------------------------------------------------------------- query


def keyword_seeds(graph: dict, question: str, k: int = 8, max_ties: int = 50) -> list[str]:
    """Keyword retrieval: score nodes by stemmed token overlap with the question
    (name + aliases), tie-broken by degree. Returns the top k, extended with every node tied
    on the best overlap count (up to max_ties): "which bearing assemblies..." must see all
    24 'bearing assembly B-1xx' nodes, not an arbitrary 8 of them."""
    q = set(content_tokens(question))
    if not q:
        return []
    deg = degrees(graph)
    scored = []
    for n in graph["nodes"]:
        toks = set(content_tokens(" ".join([n["name"], *n["aliases"]])))
        overlap = len(q & toks)
        if overlap:
            scored.append((overlap / len(toks | q) + overlap, deg[n["id"]], n["id"]))
    scored.sort(reverse=True)
    if not scored:
        return []
    best = int(scored[0][0])
    ties = [nid for s, _, nid in scored if int(s) == best][:max_ties]
    return list(dict.fromkeys([nid for *_, nid in scored[:k]] + ties))


def expand(graph: dict, seeds: Iterable[str], hops: int = 1) -> set[str]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in graph["edges"]:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
    frontier, seen = set(seeds), set(seeds)
    for _ in range(max(0, hops)):
        frontier = {m for n in frontier for m in adj[n]} - seen
        seen |= frontier
    return seen


def facts_context(graph: dict, node_ids: set[str], seeds: Iterable[str] = (), max_facts: int = 60) -> list[str]:
    """'S -RELATION-> O (chunks 1,4)' lines for edges touching a seed, inside the subgraph.
    Edges touching seeds come first so the budget is spent on what the question is about."""
    names = {n["id"]: n["name"] for n in graph["nodes"]}
    seeds = set(seeds)
    chosen = [e for e in graph["edges"] if e["source"] in node_ids and e["target"] in node_ids]
    chosen.sort(key=lambda e: (not (e["source"] in seeds or e["target"] in seeds), -e["weight"]))
    lines = []
    for e in chosen[:max_facts]:
        multi_job = len({ev["job_id"] for ev in e["evidence"]}) > 1
        refs = sorted({f"{ev['job_id']}#{ev['chunk_index']}" if multi_job else str(ev["chunk_index"])
                       for ev in e["evidence"]}, key=lambda r: _ref_sort(r if "#" in r else f"#{r}"))
        shown = ",".join(refs[:6]) + (f",+{len(refs) - 6} more" if len(refs) > 6 else "")
        lines.append(f"{names[e['source']]} -{e['relation']}-> {names[e['target']]} (chunks {shown})")
    return lines


def ask_ollama(question: str, facts: list[str], model: str, url: str, num_ctx: int = DEFAULT_NUM_CTX) -> str:
    import httpx

    payload: dict = {
        "model": model, "stream": False, "options": {"temperature": 0, "num_ctx": num_ctx},
        "messages": [
            {"role": "system", "content": "Answer the question using ONLY the knowledge-graph facts provided. "
                                          "Cite the chunk numbers in parentheses for every claim. If the facts "
                                          "do not contain the answer, say so. Be concise."},
            {"role": "user", "content": "Facts:\n" + "\n".join(f"- {f}" for f in facts) + f"\n\nQuestion: {question}"},
        ],
    }
    try:
        caps = httpx.post(f"{url}/api/show", json={"model": model}, timeout=10).json().get("capabilities", [])
        if "thinking" in caps:
            payload["think"] = False
    except Exception:
        pass
    r = httpx.post(f"{url}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    return strip_reasoning(r.json()["message"]["content"])


def strip_reasoning(text: str) -> str:
    """Some 'thinking-only' builds (e.g. qwen3:4b-2507) ignore think=false and put their
    reasoning in `content`, terminated by '</think>'. Keep only the final answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def query_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="reduce_graph.py query", description="GraphRAG question answering over graph.json")
    ap.add_argument("graph")
    ap.add_argument("question")
    ap.add_argument("--qdrant", default="none", help="URL, local path or :memory: (entity kNN); none = keyword seeds")
    ap.add_argument("--qdrant-collection-prefix", default=REQUIRED_COLLECTION_PREFIX)
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--k", type=int, default=8, help="seed entities")
    ap.add_argument("--max-facts", type=int, default=60)
    ap.add_argument("--no-llm", action="store_true", help="print retrieved facts only")
    a = ap.parse_args(argv)

    graph = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    seeds: list[str] = []
    how = "keyword"
    if a.qdrant.lower() != "none":
        emb = make_embedder(a.embed_model, a.ollama)
        if emb:
            sink = QdrantSink(open_qdrant(a.qdrant), a.qdrant_collection_prefix)
            qvec = emb.with_prefix(_nomic_prefix(a.embed_model, "search_query"))([a.question])[0]
            known = {n["id"] for n in graph["nodes"]}
            seeds = [nid for nid, _ in sink.search_entities(qvec, a.k) if nid in known]
            how = "qdrant-knn+keyword"
    # Hybrid: kNN catches paraphrases, keywords catch exhaustive "which X" lists.
    kw = keyword_seeds(graph, a.question, a.k)
    seeds = list(dict.fromkeys(seeds + kw))
    if how != "keyword" and not kw:
        how = "qdrant-knn"
    sub = expand(graph, seeds, a.hops)
    facts = facts_context(graph, sub, seeds, a.max_facts)
    names = {n["id"]: n["name"] for n in graph["nodes"]}
    print(f"seeds ({how}): {', '.join(names[s] for s in seeds) or '(none)'}")
    print(f"subgraph: {len(sub)} nodes, {len(facts)} facts used")
    for f in facts:
        print(f"  - {f}")
    if not facts:
        print("no relevant facts found in the graph")
        return 1
    if not a.no_llm:
        print("\nanswer:\n" + ask_ollama(a.question, facts, a.model, a.ollama))
    return 0


# --------------------------------------------------------------------------- reduce CLI


def reduce_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="reduce_graph.py", description="Resolve entities and build the GraphRAG graph")
    ap.add_argument("input", help="coordinator output JSONL (output_graph.jsonl)")
    ap.add_argument("--job", default=None, help="only records of this job_id")
    ap.add_argument("--out", default="graph.json")
    ap.add_argument("--graphml", default=None, help="also write GraphML here")
    ap.add_argument("--manifest", nargs="*", default=[], help="ingest.py manifest(s) for file/offset provenance")
    ap.add_argument("--qdrant", default="none", help="http(s)://host:6333 | local path | :memory: | none")
    ap.add_argument("--qdrant-collection-prefix", default=REQUIRED_COLLECTION_PREFIX,
                    help="collections are <prefix>entities and <prefix>chunks; must start with nightshift_")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model, or none")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--similarity", type=float, default=0.9)
    a = ap.parse_args(argv)

    if a.qdrant.lower() != "none":
        try:
            collection_names(a.qdrant_collection_prefix)  # fail fast before any work
        except ValueError as e:
            ap.error(str(e))

    records, load_stats = load_records(a.input, a.job)
    embedder = make_embedder(a.embed_model, a.ollama)
    res = resolve_entities(records, embedder.with_prefix(_nomic_prefix(a.embed_model, "clustering")) if embedder else None,
                           a.similarity)
    graph = build_graph(records, res, load_manifests(a.manifest))
    Path(a.out).write_text(json.dumps(graph, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.graphml:
        write_graphml(graph, a.graphml)
    print(summarize(graph, load_stats))
    for x, y, sim in res.embedding_merges[:15]:
        print(f"  embedding merge: {x!r} ~ {y!r} (cos={sim})")
    print(f"wrote {a.out}" + (f" and {a.graphml}" if a.graphml else ""))

    if a.qdrant.lower() != "none":
        if embedder is None:
            print("skipping Qdrant sink: no embedding model available (--embed-model)", file=sys.stderr)
        else:
            try:
                client = open_qdrant(a.qdrant)
            except ImportError:
                print("skipping Qdrant sink: `qdrant-client` not installed", file=sys.stderr)
                return 0
            sink = QdrantSink(client, a.qdrant_collection_prefix)
            ne = sink.upsert_entities(graph, embedder.with_prefix(_nomic_prefix(a.embed_model, "search_document")))
            nc = sink.upsert_chunks(records, res, embedder.with_prefix(_nomic_prefix(a.embed_model, "search_document")))
            print(f"qdrant {a.qdrant}: upserted {ne} -> {sink.names['entities']}, {nc} -> {sink.names['chunks']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "query":
        return query_main(argv[1:])
    if argv and argv[0] == "reduce":
        argv = argv[1:]
    return reduce_main(argv)


if __name__ == "__main__":
    sys.exit(main())
