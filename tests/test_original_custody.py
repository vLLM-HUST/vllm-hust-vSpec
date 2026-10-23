import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fetch_originals", ROOT / "scripts/fetch_benchmark_originals.py"
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def test_original_integrity():
    raw = b"synthetic integrity fixture"
    digest = hashlib.sha256(raw).hexdigest()
    assert FETCH.checked(raw, digest, len(raw)) == raw
    with pytest.raises(ValueError, match="checksum"):
        FETCH.checked(raw + b"corrupt", digest)
    with pytest.raises(ValueError, match="size"):
        FETCH.checked(raw, digest, len(raw) + 1)


@pytest.mark.parametrize("name", ["/absolute", "../parent", "a/../b", ".", "a//b"])
def test_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe"):
        FETCH.safe_path(name)


def test_external_empty_destination(tmp_path):
    assert FETCH.destination(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="external"):
        FETCH.destination(ROOT / "copy")
    (tmp_path / "existing").touch()
    with pytest.raises(ValueError, match="empty"):
        FETCH.destination(tmp_path)


def test_no_originals_return_to_git():
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    assert not tracked.intersection(config["migrated_paths"])
