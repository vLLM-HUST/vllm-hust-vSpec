#!/usr/bin/env python3
"""Check a private model-level release receipt without printing its contents.

The receipt records decisions and SHA-256 digests of private evidence. Its
contents are kept outside the repository; a maintainer can audit each digest
against the corresponding private evaluation artifact before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"MMLU-Pro", "HLE-Verified", "FrontierScience"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"model-level acceptance receipt: {reason}")


def _digest(value: object, field: str) -> None:
    _require(
        isinstance(value, str) and HEX64.fullmatch(value) is not None and value != "0" * 64,
        f"{field} must contain a nonzero SHA-256 digest",
    )


def verify_receipt(receipt: object, *, commit: str, version: str, backend_commit: str) -> None:
    """Require per-dataset model speed and quality gates bound to this source."""
    _require(isinstance(receipt, dict), "root must be an object")
    expected = {
        "schema_version",
        "plugin_version",
        "git_commit",
        "backend_commit",
        "evidence_manifest_sha256",
        "datasets",
    }
    _require(set(receipt) == expected, "unexpected or missing top-level fields")
    _require(receipt["schema_version"] == 1, "schema_version must be 1")
    _require(receipt["plugin_version"] == version, "plugin version mismatch")
    _require(
        isinstance(receipt["git_commit"], str)
        and HEX40.fullmatch(receipt["git_commit"]) is not None
        and receipt["git_commit"] == commit,
        "Git commit mismatch",
    )
    _require(receipt["backend_commit"] == backend_commit, "backend commit mismatch")
    _digest(receipt["evidence_manifest_sha256"], "evidence_manifest_sha256")
    datasets = receipt["datasets"]
    _require(
        isinstance(datasets, dict) and set(datasets) == DATASETS,
        "exactly the three required datasets must be present",
    )
    for name, row in datasets.items():
        _require(isinstance(row, dict), f"{name} must be an object")
        fields = {
            "sample_count",
            "full_split",
            "throughput_ge_1_10",
            "quality_within_2pp",
            "evidence_sha256",
        }
        _require(set(row) == fields, f"{name} has missing or unexpected fields")
        minimum = 160 if name == "FrontierScience" else 200
        _require(
            type(row["sample_count"]) is int and row["sample_count"] >= minimum,
            f"{name} sample count is below the acceptance plan",
        )
        _require(type(row["full_split"]) is bool, f"{name} full_split must be Boolean")
        if name == "FrontierScience":
            _require(row["full_split"], "FrontierScience must cover the full split")
        _require(row["throughput_ge_1_10"] is True, f"{name} throughput did not pass 1.10x")
        _require(
            row["quality_within_2pp"] is True, f"{name} quality did not pass the 2 pp tolerance"
        )
        _digest(row["evidence_sha256"], f"{name} evidence_sha256")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--receipt", type=Path, help="private JSON receipt file")
    sources.add_argument(
        "--env", metavar="NAME", help="environment variable containing JSON receipt"
    )
    parser.add_argument("--commit", help="expected 40-character release Git commit")
    args = parser.parse_args()
    if args.receipt is not None:
        raw = args.receipt.read_bytes()
    else:
        raw = os.environ.get(args.env, "").encode("utf-8")
    _require(bool(raw), "receipt is absent")
    commit = (
        args.commit
        or subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    )
    version = runpy.run_path(str(ROOT / "src/vllm_hust_vspec/_version.py"))["__version__"]
    backend_commit = runpy.run_path(str(ROOT / "src/vllm_hust_vspec/backend_bundle.py"))[
        "BASE_COMMIT"
    ]
    verify_receipt(json.loads(raw), commit=commit, version=version, backend_commit=backend_commit)
    print(f"model_level_acceptance_receipt_sha256={hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
