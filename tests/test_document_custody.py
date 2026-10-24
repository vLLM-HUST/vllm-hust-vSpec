"""Historical document entry points must retain canonical original links."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_document_projections_link_frozen_benchmark_originals():
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    projections = config["document_projections"]
    assert len(projections) == 18
    for relative, binding in projections.items():
        source = (ROOT / relative).read_bytes()
        assert binding["url"].startswith("https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/")
        assert binding["url"].endswith("/snapshot/" + relative)
        assert len(binding["sha256"]) == 64
        assert binding["url"] in source.decode()
        assert hashlib.sha256(source).hexdigest() != binding["sha256"]


def test_document_bindings_match_downloaded_original_bytes():
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    for relative, binding in config["document_projections"].items():
        assert (
            hashlib.sha256((originals / relative).read_bytes()).hexdigest() == (binding["sha256"])
        )


def test_legacy_data_links_bind_downloaded_raw_members():
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    documents = config["document_data_links"]
    assert len(documents) == 12
    for relative, rows in documents.items():
        source = (ROOT / relative).read_text()
        assert rows
        assert len({row["path"] for row in rows}) == len(rows)
        for row in rows:
            assert row["path"].startswith("benchmark_results/")
            assert "{" not in row["path"] and "}" not in row["path"]
            assert row["url"].startswith("https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/")
            assert row["url"].endswith("/snapshot/" + row["path"])
            assert row["url"] in source
            assert (
                hashlib.sha256((originals / row["path"]).read_bytes()).hexdigest() == row["sha256"]
            )


def test_reference_audit_preserves_unavailable_paths_and_scope():
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    audit = config["document_reference_audit"]
    assert audit["scope_manifest_urls"] == [c["manifest_url"] for c in config["capsules"][:6]]
    assert len(audit["scope_manifest_urls"]) == 6
    assert len(audit["documents"]) == 6
    assert sum(len(rows) for rows in audit["documents"].values()) == 23
    output_examples = []
    for relative, rows in audit["documents"].items():
        assert relative in config["document_projections"]
        source = (ROOT / relative).read_text()
        linked = {r["path"] for r in config["document_data_links"].get(relative, [])}
        assert len({row["path"] for row in rows}) == len(rows)
        for row in rows:
            assert row["path"] in source
            assert row["path"] not in linked
            assert "{" not in row["path"] and "}" not in row["path"]
            if row["status"] == "historical-output-example":
                output_examples.append(row["path"])
            else:
                assert row["status"] == "not-found-in-pinned-capsules"
                if row["path"].startswith("benchmark_results/"):
                    assert not (originals / row["path"]).exists()
    assert output_examples == ["benchmark_results/draft_b16_recovery_20260914/resume_n50"]
    assert "## 256K 正确性" in (ROOT / "docs/qwen35_frontier_mtp2.md").read_text()


def test_legacy_followup_keeps_partial_original_coverage_explicit():
    originals = Path(os.environ["VLLM_HUST_BENCHMARK_ORIGINALS"])
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    audit = config["legacy_followup_reference_audit"]
    assert audit["scope_manifest_urls"] == [c["manifest_url"] for c in config["capsules"][:7]]
    assert len(audit["documents"]) == 2
    assert sum(len(rows) for rows in audit["documents"].values()) == 12
    for relative, rows in audit["documents"].items():
        text = (ROOT / relative).read_text()
        for path in rows:
            assert path in text
            assert not (originals / path).exists()
    binding = audit["school_summary"]
    assert binding["url"].startswith(
        "https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/"
        "9ab6413d1ea47587953f037d42fea568f4a5a201/"
    )
    assert binding["url"] in (ROOT / "docs/school028_local_argmax.md").read_text()
    with urllib.request.urlopen(binding["url"], timeout=60) as response:
        raw = response.read()
    assert len(raw) == binding["bytes"]
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
