from __future__ import annotations

import os
import shutil
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


def test_release_build_removes_uv_dist_gitignore(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugin"
    bin_dir = tmp_path / "bin"
    dist_dir = plugin_dir / "dist"
    scripts_dir = plugin_dir / "scripts"
    plugin_dir.mkdir()
    bin_dir.mkdir()
    scripts_dir.mkdir()
    shutil.copy2(ROOT / "release.sh", plugin_dir / "release.sh")
    (scripts_dir / "verify_release.py").touch()

    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"#!/usr/bin/env bash\nset -euo pipefail\ntest ! -e {dist_dir / '.gitignore'}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"mkdir -p {dist_dir}\n"
        f"printf '*' > {dist_dir / '.gitignore'}\n"
        f"touch {dist_dir / 'vllm_hust_vspec-0.13.2-py3-none-any.whl'}\n"
        f"touch {dist_dir / 'vllm_hust_vspec-0.13.2.tar.gz'}\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["PYTHON_BIN"] = str(fake_python)
    subprocess.run(
        ["bash", "release.sh", "build"],
        cwd=plugin_dir,
        env=environment,
        check=True,
    )

    assert not (dist_dir / ".gitignore").exists()
