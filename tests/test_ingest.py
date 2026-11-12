"""Unit tests for ingest.py: chunking, overlap, dedupe, file handling, batching (no network)."""
import json

import pytest

import ingest


def test_approx_tokens_word_based():
    assert ingest.approx_tokens("") == 0
    assert ingest.approx_tokens("one two three") == 4  # ceil(3 * 4/3)


def test_paragraphs_kept_whole_when_they_fit():
    text = "Alpha one two.\n\nBeta three four.\n\nGamma five six."
    spans = ingest.chunk_text(text, chunk_tokens=100, overlap=0)
    assert spans == [(0, len(text))]


def test_chunks_respect_budget_and_overlap():
    paras = [f"Paragraph {i} " + " ".join(f"w{i}_{j}" for j in range(20)) + "." for i in range(10)]
    text = "\n\n".join(paras)
    spans = ingest.chunk_text(text, chunk_tokens=70, overlap=30)
    assert len(spans) > 3
    for s, e in spans:
        assert ingest.approx_tokens(text[s:e]) <= 70
    # consecutive chunks overlap (a trailing paragraph is repeated)
    assert any(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))
    # full coverage: every paragraph appears in at least one chunk
    joined = " ".join(text[s:e] for s, e in spans)
    assert all(p in joined for p in paras)


def test_long_paragraph_splits_on_sentences_not_decimals():
    text = " ".join(f"Vibration on node {i} was 4.2 mm/s at shift start." for i in range(30))
    spans = ingest.chunk_text(text, chunk_tokens=40, overlap=0)
    assert len(spans) > 1
    for s, e in spans:
        piece = text[s:e]
        assert piece.endswith("start.")  # split at sentence ends, never inside "4.2"


def test_run_on_text_falls_back_to_word_windows():
    text = " ".join(f"tok{i}" for i in range(1000))  # no punctuation at all
    spans = ingest.chunk_text(text, chunk_tokens=100, overlap=10)
    assert all(ingest.approx_tokens(text[s:e]) <= 100 for s, e in spans)
    assert text[spans[-1][0]:spans[-1][1]].endswith("tok999")


def test_chunk_files_dedupes_and_skips(tmp_path):
    (tmp_path / "a.txt").write_text("Pump P-1 failed.\n\nTechnician replaced seal.")
    (tmp_path / "b.md").write_text("Pump P-1 failed.\n\nTechnician replaced seal.")  # identical
    (tmp_path / "empty.txt").write_text("   \n")
    (tmp_path / "bin.log").write_bytes(b"abc\x00\x01def")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")  # unsupported suffix, ignored
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "x.txt").write_text("should be ignored")
    chunks, stats = ingest.chunk_files([tmp_path], chunk_tokens=100, overlap=0)
    assert len(chunks) == 1
    assert stats["duplicate_chunks"] == 1
    assert chunks[0].also_in and "b.md" in chunks[0].also_in[0]
    assert len(stats["skipped_files"]) == 2
    assert len(chunks[0].sha256) == 64


def test_json_and_jsonl_text_fields(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"id": 1, "message": "Motor M-12 tripped on overcurrent"}) + "\n{torn line\n"
        + json.dumps({"id": 2, "meta": {"note": "Belt replaced by the night technician"}, "n": 5}) + "\n")
    (tmp_path / "b.json").write_text(json.dumps({"items": [{"text": "Compressor CT-7 bearing swapped"}]}))
    chunks, _ = ingest.chunk_files([tmp_path], chunk_tokens=200, overlap=0)
    body = "\n".join(c.text for c in chunks)
    assert "Motor M-12 tripped" in body and "Belt replaced" in body and "CT-7 bearing" in body
    assert '"id"' not in body


def test_offsets_point_into_source(tmp_path):
    src = "First para here.\n\nSecond para here."
    (tmp_path / "doc.txt").write_text(src)
    chunks, _ = ingest.chunk_files([tmp_path / "doc.txt"], chunk_tokens=5, overlap=0)
    for c in chunks:
        assert src[c.start:c.end] == c.text


def test_batching_and_manifest(tmp_path):
    assert [len(b) for b in ingest.batch(list(range(12001)))] == [5000, 5000, 2001]
    chunks = [ingest.Chunk(text="t", path="p", start=0, end=1, approx_tokens=1, sha256="h")] * 3
    out = ingest.write_manifest(tmp_path, "job1", chunks)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["chunk_index"] for l in lines] == [0, 1, 2]
    assert out.name == "job1.manifest.jsonl" and lines[0]["job_id"] == "job1"


def test_dry_run_cli_submits_nothing(tmp_path, capsys, monkeypatch):
    (tmp_path / "a.txt").write_text("Pump P-1 failed. Technician replaced the seal.")
    import sys
    monkeypatch.setitem(sys.modules, "demo_submit", None)  # importing it would raise
    assert ingest.main([str(tmp_path), "--dry-run"]) == 0
    assert "would submit 1 job" in capsys.readouterr().out


def test_invalid_budget():
    with pytest.raises(ValueError):
        ingest.chunk_text("x", chunk_tokens=0)


def test_trailing_newline_and_single_line_paragraph_not_lost(tmp_path):
    src = "Header\n\n- item one\n- item two\n"
    spans = ingest.chunk_text(src, chunk_tokens=100, overlap=0)
    assert src[spans[-1][0]:spans[-1][1]].endswith("item two")
    (tmp_path / "w.csv").write_text("a,b\n1,2\n")
    chunks, _ = ingest.chunk_files([tmp_path], 100, 0)
    assert chunks and chunks[0].text == "a,b\n1,2"


def test_duplicate_jsonl_records_kept_once():
    raw = "\n".join([json.dumps({"message": "Overcurrent trip on conveyor C-12 motor"})] * 3)
    assert ingest.extract_json_text(raw, jsonl=True) == "Overcurrent trip on conveyor C-12 motor"
