#!/usr/bin/env python3
"""Validate vSpec source metadata and built release artifacts."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import runpy
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "vllm_hust_vspec"
VERSION_FILE = PACKAGE / "_version.py"
MANIFEST_DIR = PACKAGE / "manifests"
MANIFEST_NAME = "vllm-hust-extension-v0.2.json"
MANIFEST = MANIFEST_DIR / MANIFEST_NAME
EXTENSION_ID = "org.vllm-hust.vspec"
PROJECT_NAME = "vllm-hust-vspec"
WHEEL_NAME = "vllm_hust_vspec"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"release verification failed: {message}")


def source_version() -> str:
    value = runpy.run_path(str(VERSION_FILE)).get("__version__")
    require(isinstance(value, str) and bool(value), "_version.py has no version")
    return value


def verify_source() -> str:
    version = source_version()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    metadata = project["project"]
    require(metadata["name"] == PROJECT_NAME, "unexpected project name")
    require("version" in metadata["dynamic"], "project version is not dynamic")
    version_attr = project["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    require(
        version_attr == "vllm_hust_vspec._version.__version__",
        "dynamic version does not use _version.py",
    )
    package_data = project["tool"]["setuptools"]["package-data"]["vllm_hust_vspec"]
    require("manifests/*.json" in package_data, "manifest is missing from package data")
    registrations = metadata["entry-points"]["vllm_hust.extension_bundles"]
    require(
        registrations.get(EXTENSION_ID) == "vllm_hust_vspec.manifests",
        "Bundle entry point does not target the manifests package",
    )
    require((MANIFEST_DIR / "__init__.py").is_file(), "manifests package is missing")
    require(MANIFEST.is_file(), "0.2 manifest is missing")
    require(
        not (PACKAGE / MANIFEST_NAME).exists(),
        "legacy package-root manifest must be removed",
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == "0.2-experimental", "wrong schema version")
    require(manifest["extension_id"] == EXTENSION_ID, "wrong extension ID")
    require(manifest["extension_version"] == version, "manifest version mismatch")
    require(manifest["kind"] == "in_process_plugin", "wrong extension kind")
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_wheel(path: Path, version: str) -> None:
    dist_info = f"{WHEEL_NAME}-{version}.dist-info"
    required = {
        "vllm_hust_vspec/__init__.py",
        "vllm_hust_vspec/_version.py",
        "vllm_hust_vspec/online_benchmark.py",
        "vllm_hust_vspec/manifests/__init__.py",
        f"vllm_hust_vspec/manifests/{MANIFEST_NAME}",
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/entry_points.txt",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        require(required <= names, f"wheel is missing {sorted(required - names)}")
        require(
            f"vllm_hust_vspec/{MANIFEST_NAME}" not in names,
            "wheel contains the legacy package-root manifest",
        )
        metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode())
        require(metadata["Name"] == PROJECT_NAME, "wheel project name mismatch")
        require(metadata["Version"] == version, "wheel version mismatch")
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(f"{dist_info}/entry_points.txt").decode("utf-8"))
        require(
            entry_points["vllm_hust.extension_bundles"][EXTENSION_ID]
            == "vllm_hust_vspec.manifests",
            "wheel Bundle entry point mismatch",
        )
        require(
            entry_points["console_scripts"]["vllm-hust-vspec-bench"]
            == "vllm_hust_vspec.online_benchmark:main",
            "wheel ARC-Easy benchmark entry point mismatch",
        )


def verify_sdist(path: Path, version: str) -> None:
    prefix = f"{WHEEL_NAME}-{version}"
    required = {
        f"{prefix}/release.sh",
        f"{prefix}/manage.sh",
        f"{prefix}/configs/qwen25-14b-05b-arc-easy.toml",
        f"{prefix}/configs/qwen25-14b-eagle-arc-easy.toml",
        f"{prefix}/docs/arc_easy_regression.md",
        f"{prefix}/scripts/verify_release.py",
        f"{prefix}/src/vllm_hust_vspec/_version.py",
        f"{prefix}/src/vllm_hust_vspec/manifests/__init__.py",
        f"{prefix}/src/vllm_hust_vspec/manifests/{MANIFEST_NAME}",
    }
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
    require(required <= names, f"sdist is missing {sorted(required - names)}")


def verify_artifacts(dist_dir: Path, version: str) -> list[Path]:
    wheel = dist_dir / f"{WHEEL_NAME}-{version}-py3-none-any.whl"
    sdist = dist_dir / f"{WHEEL_NAME}-{version}.tar.gz"
    require(dist_dir.is_dir(), "dist directory does not exist")
    artifacts = sorted(dist_dir.iterdir())
    require(
        artifacts == sorted((wheel, sdist)),
        "dist contains stale, unrelated, or missing artifacts",
    )
    verify_wheel(wheel, version)
    verify_sdist(sdist, version)
    return [wheel, sdist]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--dist-dir", type=Path, default=ROOT / "dist")
    arguments = parser.parse_args()

    version = verify_source()
    print(f"source_version={version}")
    if arguments.source_only:
        return
    artifacts = verify_artifacts(arguments.dist_dir.resolve(), version)
    for artifact in artifacts:
        print(f"sha256={sha256(artifact)}  {artifact}")


if __name__ == "__main__":
    main()
