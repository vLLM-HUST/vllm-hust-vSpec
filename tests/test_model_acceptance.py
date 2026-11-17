from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_model_acceptance", ROOT / "scripts/verify_model_acceptance.py"
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _receipt() -> dict:
    return {
        "schema_version": 1,
        "plugin_version": "0.15.1",
        "git_commit": "a" * 40,
        "backend_commit": "b" * 40,
        "evidence_manifest_sha256": "c" * 64,
        "datasets": {
            name: {
                "sample_count": 160 if name == "FrontierScience" else 200,
                "full_split": name == "FrontierScience",
                "throughput_ge_1_10": True,
                "quality_within_2pp": True,
                "evidence_sha256": "d" * 64,
            }
            for name in GATE.DATASETS
        },
    }


def _verify(receipt: dict) -> None:
    GATE.verify_receipt(receipt, commit="a" * 40, version="0.15.1", backend_commit="b" * 40)


def test_model_receipt_requires_all_three_evidence_gates() -> None:
    receipt = _receipt()
    _verify(receipt)
    receipt["datasets"]["MMLU-Pro"]["throughput_ge_1_10"] = False
    with pytest.raises(ValueError, match="throughput"):
        _verify(receipt)
    receipt = _receipt()
    receipt["datasets"]["HLE-Verified"]["quality_within_2pp"] = False
    with pytest.raises(ValueError, match="quality"):
        _verify(receipt)


def test_model_receipt_rejects_stale_commit_and_incomplete_sampling() -> None:
    receipt = _receipt()
    receipt["git_commit"] = "e" * 40
    with pytest.raises(ValueError, match="Git commit"):
        _verify(receipt)
    receipt = _receipt()
    receipt["datasets"]["HLE-Verified"]["sample_count"] = 199
    with pytest.raises(ValueError, match="sample count"):
        _verify(receipt)
    receipt = _receipt()
    receipt["datasets"]["FrontierScience"]["full_split"] = False
    with pytest.raises(ValueError, match="full split"):
        _verify(receipt)
    receipt = _receipt()
    receipt["datasets"]["FrontierScience"]["sample_count"] = 159
    with pytest.raises(ValueError, match="sample count"):
        _verify(receipt)


def test_model_receipt_rejects_missing_evidence() -> None:
    receipt = _receipt()
    del receipt["datasets"]["MMLU-Pro"]
    with pytest.raises(ValueError, match="three required"):
        _verify(receipt)
    receipt = _receipt()
    receipt["evidence_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="nonzero SHA-256"):
        _verify(receipt)


def test_release_gate_fails_when_receipt_is_absent() -> None:
    environment = dict(os.environ)
    environment.pop("VSPEC_MODEL_ACCEPTANCE_RECEIPT", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_model_acceptance.py",
            "--env",
            "VSPEC_MODEL_ACCEPTANCE_RECEIPT",
            "--commit",
            "a" * 40,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "receipt is absent" in result.stderr
