"""Unit tests for reduce_graph.py with fake embedders and in-memory Qdrant (no network)."""
import json
import math
import zlib

import pytest

import reduce_graph as rg


def rec(task, idx, entities, triples, job="j1", text=None):
    return {"job_id": job, "task_id": task, "chunk_index": idx, "entities": entities, "triples": triples,
            "source_text": text or " ".join(entities)}


@pytest.mark.parametrize("a,b", [
    ("PLC-Node-2", "plc node 2"), ("PLC Node 2", "PLC_Node2"), ("Compressor: turbine", "compressor turbine"),
    ("The Technician", "technician"), ("4.2 mm/s", "4.2 mm: s"), ("  Bearing   Assembly B-101 ", "bearing assembly b101"),
])
def test_normalize_key_equivalences(a, b):
    assert rg.normalize_key(a) == rg.normalize_key(b)


@pytest.mark.parametrize("a,b", [("4.2 mm/s", "42 mm/s"), ("B-101", "B-102"), ("4.2 mm/s", "4.5 mm/s")])
def test_normalize_key_keeps_numbers_distinct(a, b):
    assert rg.normalize_key(a) != rg.normalize_key(b)


def test_merge_guard_digits():
    k = rg.normalize_key
    assert not rg.can_merge(k("bearing assembly B-101"), k("bearing assembly B-102"))
    assert not rg.can_merge(k("4.2 mm/s"), k("4.5 mm/s"))
    assert not rg.can_merge(k("pump 3"), k("pump"))
    assert not rg.can_merge(k("bearing assembly B-1:113"), k("bearing assembly B-113"))
    assert rg.can_merge(k("turbine vibration"), k("vibrations"))


def test_union_find():
    uf = rg.UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b"); uf.union("c", "d"); uf.union("b", "d")
    assert len({uf.find(x) for x in "abcd"}) == 1
    assert uf.find("d") == "a"  # deterministic root


def test_load_records_tolerates_torn_and_duplicate(tmp_path):
    p = tmp_path / "out.jsonl"
    lines = [json.dumps(rec("t1", 0, ["A"], [])), json.dumps(rec("t1", 0, ["B"], [])),
             json.dumps(rec("t2", 1, ["C"], [], job="j2")), '{"job_id": "j1", "task_id": "t3", "entit']
    p.write_text("\n".join(lines))
    recs, st = rg.load_records(p)
    assert [r["task_id"] for r in recs] == ["t1", "t2"] and recs[0]["entities"] == ["A"]
    assert st["bad_lines"] == 1 and st["duplicates"] == 1
    recs, st = rg.load_records(p, job="j2")
    assert [r["task_id"] for r in recs] == ["t2"] and st["filtered"] == 2


RECORDS = [
    rec("t0", 0, ["PLC-Node-2", "Technician", "bearing assembly B-101"],
        [["Technician", "REPLACED", "bearing assembly B-101"], ["Compressor turbine", "EXCEEDED", "4.2 mm/s"]]),
    rec("t1", 1, ["PLC Node 2", "technician", "bearing assembly B-102"],
        [["technician", "REPLACED", "bearing assembly B-102"], ["Compressor: turbine", "EXCEEDED", "4.2 mm: s"]]),
    rec("t2", 2, ["Technician", "Turbine vibration", "vibrations"],
        [["Technician", "REPLACED", "bearing assembly B-101"], ["Technician", "OBSERVED", "Technician"]]),
]


def test_resolve_key_only():
    res = rg.resolve_entities(RECORDS)
    n = res.nodes
    assert res.surface_to_node["PLC-Node-2"] == res.surface_to_node["PLC Node 2"]
    assert res.surface_to_node["Compressor turbine"] == res.surface_to_node["Compressor: turbine"]
    assert res.surface_to_node["bearing assembly B-101"] != res.surface_to_node["bearing assembly B-102"]
    tech = n[res.surface_to_node["technician"]]
    assert tech["name"] == "Technician" and "technician" in tech["aliases"]  # most frequent wins
    assert res.surface_to_node["Turbine vibration"] != res.surface_to_node["vibrations"]  # no embedder


class FakeEmbedder:
    """Maps texts to hand-picked vectors; everything else gets a unique orthogonal-ish vector."""

    def __init__(self, groups):
        self.groups, self.calls = groups, []

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            key = rg.normalize_key(t)
            g = next((i for i, members in enumerate(self.groups) if key in members), None)
            if g is None:
                v = [0.0] * 64
                v[zlib.crc32(key.encode()) % 60 + 4] = 1.0
            else:
                v = [0.0] * 64
                v[g] = 1.0
            out.append(v)
        return out


def test_embedding_merge_respects_threshold_and_digit_guard():
    groups = [{"turbine vibration", "vibrations"},
              {"bearing assembly b 101", "bearing assembly b 102"}]  # embedder says identical!
    emb = FakeEmbedder(groups)
    res = rg.resolve_entities(RECORDS, embedder=emb, similarity=0.9)
    assert res.surface_to_node["Turbine vibration"] == res.surface_to_node["vibrations"]
    assert res.surface_to_node["bearing assembly B-101"] != res.surface_to_node["bearing assembly B-102"]
    embedded = {rg.normalize_key(t) for call in emb.calls for t in call}
    assert "bearing assembly b 102" not in embedded  # guard prunes before embedding
    # threshold above any similarity => no merges
    res2 = rg.resolve_entities(RECORDS, embedder=FakeEmbedder(groups), similarity=1.01)
    assert not res2.embedding_merges


def test_blocking_skips_oversized_blocks():
    keys = [f"widget {c}" for c in "abcdefghij"]
    assert rg.candidate_pairs(keys, max_block=50)
    assert not rg.candidate_pairs(keys, max_block=5)


def test_build_graph_dedupes_edges_with_evidence():
    res = rg.resolve_entities(RECORDS)
    g = rg.build_graph(RECORDS, res)
    tech = res.surface_to_node["Technician"]
    b101 = res.surface_to_node["bearing assembly B-101"]
    e = next(e for e in g["edges"] if e["source"] == tech and e["target"] == b101)
    assert e["weight"] == 2 and [ev["chunk_index"] for ev in e["evidence"]] == [0, 2]
    exceeded = [e for e in g["edges"] if e["relation"] == "EXCEEDED"]
    assert len(exceeded) == 1 and exceeded[0]["weight"] == 2
    assert g["stats"]["self_loops_dropped"] == 1
    node = next(n for n in g["nodes"] if n["id"] == tech)
    assert node["mention_count"] == 3 and node["chunks"] == ["j1#0", "j1#1", "j1#2"]
    assert "Technician" in rg.summarize(g)


def test_manifest_provenance_and_graphml(tmp_path):
    m = tmp_path / "j1.manifest.jsonl"
    m.write_text(json.dumps({"job_id": "j1", "chunk_index": 0, "path": "a.txt", "start": 0, "end": 9, "sha256": "x"}))
    g = rg.build_graph(RECORDS, rg.resolve_entities(RECORDS), rg.load_manifests([m]))
    assert g["chunks"]["j1#0"]["source"]["path"] == "a.txt"
    out = tmp_path / "g.graphml"
    rg.write_graphml(g, out)
    assert out.read_text().count("<node ") == len(g["nodes"])


def test_query_helpers():
    g = rg.build_graph(RECORDS, rg.resolve_entities(RECORDS))
    seeds = rg.keyword_seeds(g, "which bearing assemblies were replaced?")
    names = {n["id"]: n["name"] for n in g["nodes"]}
    assert {names[s] for s in seeds[:2]} == {"bearing assembly B-101", "bearing assembly B-102"}
    sub = rg.expand(g, seeds, hops=1)
    facts = rg.facts_context(g, sub, seeds)
    assert any("REPLACED-> bearing assembly B-101 (chunks 0,2)" in f for f in facts)
    assert rg.strip_reasoning("hmm let me think</think>\n\nB-101 (0)") == "B-101 (0)"


def test_collection_prefix_guard():
    with pytest.raises(ValueError):
        rg.collection_names("venues_")
    with pytest.raises(ValueError):
        rg.QdrantSink(object(), "events")
    with pytest.raises(ValueError):
        rg._check_collection_name("venues")
    assert rg.collection_names("nightshift_test_")["entities"] == "nightshift_test_entities"


def test_qdrant_memory_upsert_idempotent_and_search():
    pytest.importorskip("qdrant_client")
    client = rg.open_qdrant(":memory:")
    sink = rg.QdrantSink(client, "nightshift_unittest_")
    res = rg.resolve_entities(RECORDS)
    g = rg.build_graph(RECORDS, res)
    emb = FakeEmbedder([])
    for _ in range(2):  # re-run must not duplicate points
        assert sink.upsert_entities(g, emb) == len(g["nodes"])
        assert sink.upsert_chunks(RECORDS, res, emb) == 3
    assert client.count("nightshift_unittest_entities").count == len(g["nodes"])
    assert client.count("nightshift_unittest_chunks").count == 3
    target = g["nodes"][0]
    hits = sink.search_entities(emb([rg.entity_text(target)])[0], 1)
    assert hits[0][0] == target["id"]


def test_qdrant_incompatible_collection_fails_without_recreate():
    pytest.importorskip("qdrant_client")
    from qdrant_client import models
    client = rg.open_qdrant(":memory:")
    client.create_collection("nightshift_x_entities", vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
    sink = rg.QdrantSink(client, "nightshift_x_")
    called = []
    client.delete_collection = lambda *a, **k: called.append("delete")  # must never be used
    client.recreate_collection = lambda *a, **k: called.append("recreate")
    g = rg.build_graph(RECORDS, rg.resolve_entities(RECORDS))
    with pytest.raises(RuntimeError, match="Refusing to recreate"):
        sink.upsert_entities(g, FakeEmbedder([]))
    assert not called


def test_cosine():
    assert math.isclose(rg.cosine([1, 0], [1, 0]), 1.0)
    assert rg.cosine([1, 0], [0, 0]) == 0.0
