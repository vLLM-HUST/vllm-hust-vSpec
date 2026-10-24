"""Historical document entry points must retain canonical original links."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_document_projections_link_frozen_benchmark_originals():
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    projections = config["document_projections"]
    assert len(projections) == 4
    for relative, binding in projections.items():
        source = (ROOT / relative).read_bytes()
        assert binding["url"].startswith(
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
        )
        assert binding["url"].endswith("/snapshot/" + relative)
        assert len(binding["sha256"]) == 64
        assert binding["url"] in source.decode()
        assert hashlib.sha256(source).hexdigest() != binding["sha256"]
