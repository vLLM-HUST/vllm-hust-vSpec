from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_source_metadata_is_consistent() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_release.py", "--source-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source_version=0.13.2" in result.stdout
