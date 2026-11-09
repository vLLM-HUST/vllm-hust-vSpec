"""Install the matched Qwen3.5 Ascend backend and vSpec operators in one run.

The host driver, CANN SDK, pinned Python stack and model are prerequisites.
Generated sources, wheels and device proofs are kept in private storage.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ._version import __version__
from .backend_bundle import BASE_COMMIT

OPERATORS = (
    ("conv-reference", "validate_conv", ("--component", "reference")),
    ("conv-stride", "validate_conv", ("--component", "stride")),
    ("conv-edge", "validate_conv", ("--component", "edge")),
    ("gamma6-tail", "validate_tail_conv", ()),
    ("gamma6-rgdr-reference", "validate_rgdr", ("--component", "gamma6-reference")),
    ("gamma6-rgdr-packed-reference", "validate_rgdr", ("--component", "gamma6-packed-reference")),
    ("gamma6-rgdr", "validate_rgdr", ("--component", "gamma6-packed")),
    ("gamma8-rgdr", "validate_rgdr", ("--component", "gamma8")),
    ("rgdr-state-pipeline", "validate_rgdr_state_pipeline", ()),
    ("fia-host", "validate_fia_host", ()),
)

EXPORTS = {
    "VSPEC_NATIVE_STRIDE_CONV_DIR": "conv-stride",
    "VSPEC_EDGE_CONV_DIR": "conv-edge",
    "VSPEC_GAMMA6_CONV_DIR": "gamma6-tail",
    "VSPEC_GAMMA6_RGDR_DIR": "gamma6-rgdr",
    "VSPEC_GAMMA8_RGDR_DIR": "gamma8-rgdr",
    "VSPEC_RGDR_ARTIFACT_DIR": "rgdr-state-pipeline",
    "VSPEC_NATIVE_FIA_ARTIFACT_DIR": "fia-host",
    "VSPEC_GAMMA6_INPUT_DIR": "gdn-input-6",
    "VSPEC_GAMMA8_INPUT_DIR": "gdn-input-8",
    "VSPEC_GAMMA6_MASK_DIR": "mask",
    "VSPEC_PREFIX_CLIP_BRIDGE_DIR": "prefix-clip",
    "VSPEC_TWO_BANK_EMBEDDING_DIR": "two-bank",
    "VSPEC_INPLACE_HEAD_NZ_DIR": "inplace-head",
}


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--wheel", type=Path, help="Plugin wheel; optional when already installed")
    parser.add_argument(
        "--ascend-source",
        required=True,
        type=Path,
        help=f"Git checkout containing backend commit {BASE_COMMIT}",
    )
    parser.add_argument(
        "--vllm-source", required=True, type=Path, help="Pinned vLLM source checkout"
    )
    parser.add_argument("--cann-root", required=True, type=Path, help="Installed CANN SDK root")
    parser.add_argument("--model-dir", required=True, type=Path, help="Read-only model checkpoint")
    parser.add_argument(
        "--private-dir",
        required=True,
        type=Path,
        help="New private directory for backend and component proofs",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Show install plan without writes")
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The matched Qwen3.5 runtime requires Python 3.11")
    required_versions = {"torch": "2.13.0", "torch-npu": "2.13.0rc1", "triton-ascend": "3.6.0"}
    for package, version in required_versions.items():
        installed = importlib.metadata.version(package)
        if not installed.startswith(version):
            raise RuntimeError(f"{package} must start with {version}; found {installed}")
    if "g762f85b31" not in importlib.metadata.version("vllm"):
        raise RuntimeError("Install the pinned vLLM-HUST revision before vSpec")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    args.private_dir = args.private_dir.expanduser().resolve()
    for name in ("ascend_source", "vllm_source", "cann_root", "model_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve(strict=True))
    if args.plugin_source is not None:
        args.plugin_source = args.plugin_source.resolve(strict=True)
    if args.wheel is not None:
        args.wheel = args.wheel.resolve(strict=True)
        if not args.wheel.is_file():
            raise FileNotFoundError(args.wheel)
    elif (
        args.plugin_source is None and importlib.metadata.version("vllm-hust-vspec") != __version__
    ):
        raise RuntimeError("Installed plugin does not match this installer")
    if args.private_dir.exists() or args.private_dir.is_symlink():
        raise FileExistsError(f"Use a new private directory: {args.private_dir}")
    if (
        args.private_dir.is_relative_to(args.model_dir)
        or args.private_dir.is_relative_to(args.ascend_source)
        or args.private_dir.is_relative_to(args.vllm_source)
        or str(args.private_dir).startswith("/data/shared_upload/")
    ):
        raise ValueError("Builds and proofs must be in private storage outside model/source trees")
    if not (args.cann_root / "aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake").is_file():
        raise FileNotFoundError("CANN AscendC CMake support is missing")
    if not (args.model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("Model checkpoint index is missing")
    if not (args.ascend_source / "vllm_ascend/attention/attention_v1.py").is_file():
        raise FileNotFoundError("Ascend backend source is missing")
    if not (args.vllm_source / "vllm").is_dir():
        raise FileNotFoundError("vLLM source package is missing")
    subprocess.run(
        ["git", "-C", str(args.ascend_source), "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    parent = args.private_dir.parent
    while not parent.exists():
        parent = parent.parent
    if not args.dry_run and shutil.disk_usage(parent).free < 15 * 1024**3:
        raise RuntimeError("Private storage needs at least 15 GiB free")
    if not args.dry_run:
        info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
        if info.count("No running processes found") != 2:
            raise RuntimeError("Both allocated NPUs must be idle during operator validation")


def module(name: str, *options: object) -> list[str]:
    return [sys.executable, "-m", f"vllm_hust_vspec.{name}", *(str(v) for v in options)]


def commands(args: argparse.Namespace, backend_site: Path) -> list[tuple[str, list[str]]]:
    root = args.private_dir
    artifact = root / "components"
    steps: list[tuple[str, list[str]]] = []

    def add(name: str, command: list[str]) -> None:
        steps.append((name, command))

    if args.wheel is not None:
        add("plugin", [sys.executable, "-m", "pip", "install", "--no-deps", str(args.wheel)])
    elif args.plugin_source is not None:
        add(
            "plugin",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--editable",
                str(args.plugin_source),
            ],
        )
    else:
        add("plugin-installed", [sys.executable, "-c", "import vllm_hust_vspec"])
    add(
        "backend-source",
        module(
            "backend_bundle",
            "--source",
            args.ascend_source,
            "--destination",
            root / "backend-source",
        ),
    )
    add(
        "backend-wheel",
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(root / "backend-wheels"),
            str(root / "backend-source"),
        ],
    )
    for operator, validator, options in OPERATORS:
        directory = artifact / operator
        add(
            f"build-{operator}",
            module(
                "native_operator_build",
                "--operator",
                operator,
                "--cann-root",
                args.cann_root,
                "--destination",
                directory,
                "--jobs",
                args.jobs,
            ),
        )
        validation = list(options)
        if operator == "fia-host":
            validation += ["--directory", str(directory)]
        else:
            validation += [
                "--component-dir",
                str(directory),
                "--backend-package",
                str(backend_site),
            ]
            if operator == "conv-stride":
                validation += ["--reference-dir", str(artifact / "conv-reference")]
            elif operator == "gamma6-rgdr-packed-reference":
                validation += ["--reference-dir", str(artifact / "gamma6-rgdr-reference")]
            elif operator == "gamma6-rgdr":
                validation += [
                    "--reference-dir",
                    str(artifact / "gamma6-rgdr-reference"),
                    "--prior-dir",
                    str(artifact / "gamma6-rgdr-packed-reference"),
                ]
        add(f"validate-{operator}", module(validator, *validation))
    for width in (6, 8):
        add(
            f"validate-gdn-{width}",
            module(
                "validate_gdn_input",
                "--width",
                width,
                "--backend",
                backend_site,
                "--vllm-source",
                args.vllm_source,
                "--destination",
                artifact / f"gdn-input-{width}",
            ),
        )
    for label, name in (("mask", "validate_mask"), ("prefix-clip", "validate_prefix_clip")):
        add(
            f"validate-{label}",
            module(name, "--backend", backend_site, "--destination", artifact / label),
        )
    for label, name in (
        ("two-bank", "validate_two_bank"),
        ("inplace-head", "validate_inplace_head"),
    ):
        add(
            f"validate-{label}",
            module(name, "--model", args.model_dir, "--destination", artifact / label),
        )
    return steps


def installed_backend_site() -> Path:
    distribution = importlib.metadata.distribution("vllm-ascend")
    site = Path(distribution.locate_file(".")).resolve()
    vendor = (
        site / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
    )
    if not vendor.is_file():
        raise FileNotFoundError(f"Installed backend custom operator is missing: {vendor}")
    return site


def run(args: argparse.Namespace) -> Path:
    preflight(args)
    root = args.private_dir
    if args.dry_run:
        site = Path(sys.prefix) / "lib/python3.11/site-packages"
        for label, command in commands(args, site):
            print(f"{label}: {shlex.join(command)}")
        print("backend-install: python -m pip install --no-deps <built-backend-wheel>")
        print(f"runtime-env: {root / 'runtime.env'}")
        return root
    os.umask(0o077)
    root.mkdir(parents=True, mode=0o700)
    (root / "backend-wheels").mkdir(mode=0o700)
    environment = dict(os.environ, ASCEND_HOME_PATH=str(args.cann_root))
    if args.plugin_source is not None:
        bootstrap = str(args.plugin_source / "src")
        source_path = environment.get("PYTHONPATH", "").split(os.pathsep)
        environment["PYTHONPATH"] = os.pathsep.join(p for p in source_path if p != bootstrap)
    for label, command in commands(args, Path(sys.prefix))[0:3]:
        print(f"Installing {label}", flush=True)
        subprocess.run(command, check=True, env=environment)
        if (
            label.startswith("plugin")
            and importlib.metadata.version("vllm-hust-vspec") != __version__
        ):
            raise RuntimeError("Installed plugin version does not match the installer")
    wheels = sorted((root / "backend-wheels").glob("vllm_ascend*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one built backend wheel, found {len(wheels)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheels[0])],
        check=True,
        env=environment,
    )
    site = installed_backend_site()
    for label, command in commands(args, site)[3:]:
        print(f"Installing {label}", flush=True)
        subprocess.run(command, check=True, env=environment)
    artifact = root / "components"
    exports = {key: str(artifact / value) for key, value in EXPORTS.items()}
    exports.update(
        VSPEC_ASCEND_SOURCE_ROOT=str(site),
        VSPEC_VLLM_SOURCE_ROOT=str(args.vllm_source),
        VSPEC_NATIVE_FIA_ORACLE_SOURCE=str(
            args.ascend_source / "vllm_ascend/attention/attention_v1.py"
        ),
    )
    env_file = root / "runtime.env"
    env_file.write_text(
        "".join(f"export {key}={shlex.quote(value)}\n" for key, value in sorted(exports.items()))
    )
    env_file.chmod(0o600)
    receipt = root / "installed.json"
    receipt.write_text(
        json.dumps(
            {
                "plugin_version": importlib.metadata.version("vllm-hust-vspec"),
                "backend_wheel": str(wheels[0]),
                "backend_site": str(site),
                "runtime_env": str(env_file),
                "components": exports,
            },
            indent=2,
        )
        + "\n"
    )
    receipt.chmod(0o600)
    print(f"Complete. Source {env_file} before launching vSpec.")
    return root


def main() -> None:
    run(arguments())


if __name__ == "__main__":
    main()
