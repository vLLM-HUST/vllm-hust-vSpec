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
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "vllm_hust_vspec"
VERSION_FILE = PACKAGE / "_version.py"
MANIFEST_DIR = PACKAGE / "manifests"
MANIFEST_NAME = "vllm-hust-extension-v0.3.json"
MANIFEST = MANIFEST_DIR / MANIFEST_NAME
EXTENSION_ID = "org.vllm-hust.vspec"
EVALUATION_ENDPOINT = "evaluation_endpoint:EvaluationEndpointPlugin"
PROJECT_NAME = "vllm-hust-vspec"
WHEEL_NAME = "vllm_hust_vspec"
HOST_VERSION_RANGE = ">=0.17.2rc1.dev5871,<0.26"
REQUIRED_TOOLS = {
    "vllm-hust-vspec-prepare-backend": "vllm_hust_vspec.backend_bundle:main",
    "vllm-hust-vspec-build-operator": "vllm_hust_vspec.native_operator_build:main",
    "vllm-hust-vspec-run-with-backend": "vllm_hust_vspec.run_with_backend:main",
    "vllm-hust-vspec-validate-conv": "vllm_hust_vspec.validate_conv:main",
    "vllm-hust-vspec-validate-tail-conv": "vllm_hust_vspec.validate_tail_conv:main",
    "vllm-hust-vspec-validate-rgdr": "vllm_hust_vspec.validate_rgdr:main",
    "vllm-hust-vspec-validate-rgdr-state-pipeline": (
        "vllm_hust_vspec.validate_rgdr_state_pipeline:main"
    ),
    "vllm-hust-vspec-validate-gdn-input": "vllm_hust_vspec.validate_gdn_input:main",
    "vllm-hust-vspec-validate-mask": "vllm_hust_vspec.validate_mask:main",
    "vllm-hust-vspec-validate-prefix-clip": "vllm_hust_vspec.validate_prefix_clip:main",
    "vllm-hust-vspec-validate-fia-host": "vllm_hust_vspec.validate_fia_host:main",
    "vllm-hust-vspec-validate-two-bank": "vllm_hust_vspec.validate_two_bank:main",
    "vllm-hust-vspec-validate-inplace-head": "vllm_hust_vspec.validate_inplace_head:main",
}
REQUIRED_VENDOR = {
    "_vendor/ascend_hust/build-python.patch",
    "_vendor/ascend_hust/tracked-changes.patch",
    "_vendor/runtime/campaign_candidate_worker.py",
    "_vendor/runtime/preflight.py",
    "_vendor/native/conv_reference/CMakeLists.txt",
    "_vendor/native/conv_stride/CMakeLists.txt",
    "_vendor/native/conv_edge/CMakeLists.txt",
    "_vendor/native/gamma6_tail/CMakeLists.txt",
    "_vendor/native/gamma6_rgdr_reference/CMakeLists.txt",
    "_vendor/native/gamma6_rgdr_packed_reference/CMakeLists.txt",
    "_vendor/native/gamma6_rgdr/CMakeLists.txt",
    "_vendor/native/gamma8_rgdr/CMakeLists.txt",
    "_vendor/native/rgdr_state_pipeline/CMakeLists.txt",
    "_vendor/native/fia_host/CMakeLists.txt",
    "_vendor/operators/gamma6_input/kernel.py",
    "_vendor/operators/gamma8_input/kernel.py",
    "_vendor/operators/gamma6_mask/kernel.py",
    "_vendor/operators/prefix_clip/clip_kernel.py",
    "_vendor/operators/two_bank_embedding/kernel.py",
    "_vendor/operators/inplace_head_nz/real_weight.py",
    "_vendor/qwen35_mtp_compat/sitecustomize.py",
    "_vendor/qwen35_mtp_compat/qwen35_mtp_kv_compat.py",
    "_vendor/qwen35_mtp_compat/qwen35_mtp_mamba_compat.py",
}


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
    require(
        metadata["entry-points"].get("vllm.endpoint_plugins", {}).get("vspec_eval")
        == EVALUATION_ENDPOINT,
        "evaluation endpoint is not registered",
    )
    scripts = metadata["scripts"]
    require(
        "vllm-hust-vspec-models" not in scripts,
        "release must not expose an automatic model download command",
    )
    for name, entry in REQUIRED_TOOLS.items():
        require(scripts.get(name) == entry, f"missing or incorrect release tool: {name}")
    for asset in REQUIRED_VENDOR:
        require((PACKAGE / asset).is_file(), f"required vendor source missing: {asset}")
    require((ROOT / "docs/v0.15.1-reproduction.md").is_file(), "reproduction guide is missing")
    for relative in (
        "evaluation/evaluation_endpoint.py",
        "evaluation/evaluation_worker_common.py",
        "evaluation/dynamic_worker.py",
        "evaluation_static/native_static_worker.py",
        "evaluation_static/capture_safe_static.py",
        "evaluation_static/hot_static_options.py",
        "evaluation_static/static_install_metadata.py",
        "evaluation_static/vspec_readonly_observer.py",
        "evaluation_static/vspec_readonly_compiler.py",
        "evaluation_static/vspec_readonly_nge_worker.py",
        "evaluation_static/vspec_readonly_static_worker.py",
    ):
        require(
            (PACKAGE / relative).is_file(),
            f"model-level evaluation source missing: {relative}",
        )
    require(
        (ROOT / "scripts/verify_model_acceptance.py").is_file(),
        "model-level release gate is missing",
    )
    require((MANIFEST_DIR / "__init__.py").is_file(), "manifests package is missing")
    require(MANIFEST.is_file(), "0.3 manifest is missing")
    require(
        not (PACKAGE / MANIFEST_NAME).exists(),
        "legacy package-root manifest must be removed",
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == "0.3-experimental", "wrong schema version")
    require(manifest["extension_id"] == EXTENSION_ID, "wrong extension ID")
    require(manifest["extension_version"] == version, "manifest version mismatch")
    require(manifest["kind"] == "in_process_plugin", "wrong extension kind")
    require(manifest["host"]["provider"] == "vllm", "wrong host provider")
    require(manifest["host"]["name"] == "vllm", "wrong host distribution")
    require(
        manifest["host"]["version_range"] == HOST_VERSION_RANGE,
        "wrong host admission range",
    )
    return version


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vendor_payload() -> set[str]:
    """Require every packaged source asset, not only a historical sample."""
    vendor = PACKAGE / "_vendor"
    return {
        f"vllm_hust_vspec/{path.relative_to(PACKAGE).as_posix()}"
        for path in vendor.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def verify_package_bytes(relative: str, content: bytes, artifact: str) -> None:
    parts = PurePosixPath(relative).parts
    require(
        bool(parts) and parts[0] != "/" and ".." not in parts,
        f"{artifact} has invalid package path: {relative}",
    )
    source = PACKAGE.joinpath(*parts)
    require(source.is_file(), f"{artifact} package file is missing from source: {relative}")
    require(
        content == source.read_bytes(),
        f"{artifact} package file differs from source: {relative}",
    )


def verify_wheel(path: Path, version: str) -> None:
    dist_info = f"{WHEEL_NAME}-{version}.dist-info"
    required = {
        "vllm_hust_vspec/__init__.py",
        "vllm_hust_vspec/_version.py",
        "vllm_hust_vspec/backends/mtp.py",
        "vllm_hust_vspec/cli.py",
        "vllm_hust_vspec/online_benchmark.py",
        "vllm_hust_vspec/backend_bundle.py",
        "vllm_hust_vspec/native_operator_build.py",
        "vllm_hust_vspec/validate_rgdr_state_pipeline.py",
        "vllm_hust_vspec/run_with_backend.py",
        "vllm_hust_vspec/evaluation/evaluation_endpoint.py",
        "vllm_hust_vspec/evaluation/evaluation_worker_common.py",
        "vllm_hust_vspec/evaluation/dynamic_worker.py",
        "vllm_hust_vspec/evaluation_static/native_static_worker.py",
        "vllm_hust_vspec/evaluation_static/capture_safe_static.py",
        "vllm_hust_vspec/evaluation_static/hot_static_options.py",
        "vllm_hust_vspec/evaluation_static/static_install_metadata.py",
        "vllm_hust_vspec/evaluation_static/vspec_readonly_observer.py",
        "vllm_hust_vspec/evaluation_static/vspec_readonly_compiler.py",
        "vllm_hust_vspec/evaluation_static/vspec_readonly_nge_worker.py",
        "vllm_hust_vspec/evaluation_static/vspec_readonly_static_worker.py",
        "vllm_hust_vspec/manifests/__init__.py",
        f"vllm_hust_vspec/manifests/{MANIFEST_NAME}",
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/entry_points.txt",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required.update(vendor_payload())
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
            entry_points["vllm.endpoint_plugins"]["vspec_eval"] == EVALUATION_ENDPOINT,
            "wheel evaluation endpoint entry point mismatch",
        )
        require(
            entry_points["console_scripts"]["vllm-hust-vspec-bench"]
            == "vllm_hust_vspec.online_benchmark:main",
            "wheel ARC-Easy benchmark entry point mismatch",
        )
        for name, entry in REQUIRED_TOOLS.items():
            require(
                entry_points["console_scripts"].get(name) == entry,
                f"wheel release tool entry point mismatch: {name}",
            )
        package_prefix = f"{WHEEL_NAME}/"
        for member in archive.infolist():
            if member.filename.startswith(package_prefix) and not member.is_dir():
                verify_package_bytes(
                    member.filename.removeprefix(package_prefix),
                    archive.read(member),
                    "wheel",
                )


def verify_sdist(path: Path, version: str) -> None:
    prefix = f"{WHEEL_NAME}-{version}"
    required = {
        f"{prefix}/release.sh",
        f"{prefix}/manage.sh",
        f"{prefix}/configs/agentx-qwen35-frontier-mtp2.example.json",
        f"{prefix}/configs/agentx-qwen35-frontier-target-only.example.json",
        f"{prefix}/configs/qwen25-14b-05b-arc-easy.toml",
        f"{prefix}/configs/qwen25-14b-eagle-arc-easy.toml",
        f"{prefix}/configs/qwen35-35b-a3b-frontier-mtp2.toml",
        f"{prefix}/docs/arc_easy_regression.md",
        f"{prefix}/docs/agentx_256k_benchmark.md",
        f"{prefix}/docs/v0.15.1-reproduction.md",
        f"{prefix}/examples/qwen35_three_datasets/README.md",
        f"{prefix}/examples/qwen35_three_datasets/select_cases.py",
        f"{prefix}/examples/qwen35_three_datasets/paired.py",
        f"{prefix}/docs/qwen35_frontier_mtp2.md",
        f"{prefix}/docs/qwen35_mtp2_b16_graph_benchmark_commands.md",
        f"{prefix}/scripts/benchmark_qwen35_frontier_mtp2_gsm8k.sh",
        f"{prefix}/scripts/verify_release.py",
        f"{prefix}/scripts/verify_model_acceptance.py",
        f"{prefix}/src/vllm_hust_vspec/_version.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation/evaluation_endpoint.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation/evaluation_worker_common.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation/dynamic_worker.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/native_static_worker.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/capture_safe_static.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/hot_static_options.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/static_install_metadata.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/vspec_readonly_observer.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/vspec_readonly_compiler.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/vspec_readonly_nge_worker.py",
        f"{prefix}/src/vllm_hust_vspec/evaluation_static/vspec_readonly_static_worker.py",
        f"{prefix}/src/vllm_hust_vspec/manifests/__init__.py",
        f"{prefix}/src/vllm_hust_vspec/manifests/{MANIFEST_NAME}",
    }
    with tarfile.open(path, "r:gz") as archive:
        names = set(archive.getnames())
        required.update({f"{prefix}/src/{asset}" for asset in vendor_payload()})
        require(required <= names, f"sdist is missing {sorted(required - names)}")
        package_prefix = f"{prefix}/src/{WHEEL_NAME}/"
        for member in archive.getmembers():
            if not member.name.startswith(package_prefix) or member.isdir():
                continue
            require(member.isfile(), f"sdist has non-file package member: {member.name}")
            stream = archive.extractfile(member)
            require(stream is not None, f"sdist cannot read package file: {member.name}")
            with stream:
                verify_package_bytes(
                    member.name.removeprefix(package_prefix), stream.read(), "sdist"
                )


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
