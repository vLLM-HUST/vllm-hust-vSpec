from __future__ import annotations

import configparser
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "vllm_hust_vspec"
VERSION = "0.15.1"
STALE_FILE = "adaptive/runtime.py"


def _package_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _package_content(source: Path, relative: str, stale: bool) -> bytes:
    content = source.read_bytes()
    if stale and relative == STALE_FILE:
        assert content
        return bytes([content[0] ^ 1]) + content[1:]
    return content


def _write_test_artifacts(dist_dir: Path, stale_artifact: str | None = None) -> None:
    dist_dir.mkdir()
    package_files = _package_files()
    wheel = dist_dir / f"vllm_hust_vspec-{VERSION}-py3-none-any.whl"
    dist_info = f"vllm_hust_vspec-{VERSION}.dist-info"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    entry_points = configparser.ConfigParser()
    entry_points.read_dict(project["entry-points"])
    entry_points["console_scripts"] = project["scripts"]
    entry_points_text = io.StringIO()
    entry_points.write(entry_points_text)
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in package_files:
            relative = source.relative_to(PACKAGE).as_posix()
            content = _package_content(source, relative, stale_artifact == "wheel")
            archive.writestr(f"vllm_hust_vspec/{relative}", content)
        archive.writestr(
            f"{dist_info}/METADATA", f"Name: vllm-hust-vspec\nVersion: {VERSION}\n"
        )
        archive.writestr(f"{dist_info}/RECORD", "")
        archive.writestr(f"{dist_info}/entry_points.txt", entry_points_text.getvalue())

    prefix = f"vllm_hust_vspec-{VERSION}"
    sdist = dist_dir / f"{prefix}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for relative in (
            "release.sh",
            "manage.sh",
            "configs",
            "docs",
            "examples/qwen35_three_datasets",
            "scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh",
            "scripts/verify_release.py",
            "scripts/verify_model_acceptance.py",
        ):
            archive.add(ROOT / relative, arcname=f"{prefix}/{relative}")
        for source in package_files:
            relative = source.relative_to(PACKAGE).as_posix()
            content = _package_content(source, relative, stale_artifact == "sdist")
            member = tarfile.TarInfo(f"{prefix}/src/vllm_hust_vspec/{relative}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _verify_dist(dist_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/verify_release.py", "--dist-dir", str(dist_dir)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_release_source_metadata_is_consistent() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_release.py", "--source-only"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "source_version=0.15.1" in result.stdout


def test_release_accepts_matching_runtime_package_bytes(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    _write_test_artifacts(dist_dir)

    result = _verify_dist(dist_dir)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sha256=") == 2


@pytest.mark.parametrize("stale_artifact", ["wheel", "sdist"])
def test_release_rejects_stale_runtime_package_bytes(
    tmp_path: Path, stale_artifact: str
) -> None:
    dist_dir = tmp_path / "dist"
    _write_test_artifacts(dist_dir, stale_artifact)

    result = _verify_dist(dist_dir)

    assert result.returncode != 0
    assert (
        f"release verification failed: {stale_artifact} package file differs from source: "
        f"{STALE_FILE}"
    ) in result.stderr


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
        f"touch {dist_dir / 'vllm_hust_vspec-0.15.1-py3-none-any.whl'}\n"
        f"touch {dist_dir / 'vllm_hust_vspec-0.15.1.tar.gz'}\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    environment = os.environ.copy()
    environment.pop("BASH_ENV", None)
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["PYTHON_BIN"] = str(fake_python)
    subprocess.run(
        ["bash", "release.sh", "build"],
        cwd=plugin_dir,
        env=environment,
        check=True,
    )

    assert not (dist_dir / ".gitignore").exists()
