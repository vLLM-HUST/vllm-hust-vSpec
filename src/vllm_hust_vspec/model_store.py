"""Install-time discovery and runtime resolution of bundled default models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_REGISTRY_ENV = "HUST_VSPEC_MODEL_REGISTRY"
MODEL_DIR_ENV = "HUST_VSPEC_MODEL_DIR"
REGISTRY_SCHEMA_VERSION = 1
SHARED_MODEL_DIR = Path("/data/shared-models")
CONTAINER_MODEL_DIR = Path("/model")


class ModelStoreError(ValueError):
    """Raised when a configured or downloaded default model is unusable."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    method: str
    repo_id: str
    revision: str
    directory_name: str
    override_environment: str
    architectures: tuple[str, ...]
    requires_tokenizer: bool


DEFAULT_MODEL_SPECS = (
    ModelSpec(
        key="draft",
        method="draft_model",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        directory_name="Qwen2.5-0.5B-Instruct",
        override_environment="HUST_VSPEC_DRAFT_MODEL",
        architectures=("Qwen2ForCausalLM",),
        requires_tokenizer=True,
    ),
    ModelSpec(
        key="eagle",
        method="eagle",
        repo_id="Zjcxy-SmartAI/Eagle-Qwen2.5-14B-Instruct",
        revision="b8d3782e850e7692ea4e5c833e8477743998b706",
        directory_name="Eagle-Qwen2.5-14B-Instruct",
        override_environment="HUST_VSPEC_EAGLE_MODEL",
        architectures=("Qwen2ForCausalLMEagle",),
        requires_tokenizer=False,
    ),
)


def default_registry_path(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get(MODEL_REGISTRY_ENV)
    if configured:
        return Path(configured).expanduser()
    config_home = values.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "vllm-hust-vspec" / "models.json"


def default_model_dir(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get(MODEL_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    shared = SHARED_MODEL_DIR
    if shared.is_dir() and os.access(shared, os.W_OK):
        return shared
    data_home = values.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return root / "vllm-hust-vspec" / "models"


def _spec_for_method(method: str) -> ModelSpec | None:
    return next((spec for spec in DEFAULT_MODEL_SPECS if spec.method == method), None)


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelStoreError(f"cannot read model registry {path}: {exc}") from exc
    if document.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ModelStoreError(f"unsupported model registry schema in {path}")
    models = document.get("models")
    if not isinstance(models, dict):
        raise ModelStoreError(f"model registry {path} has no models object")
    return models


def _registry_model_path(models: Mapping[str, Any], key: str) -> Path | None:
    entry = models.get(key)
    if isinstance(entry, str):
        return Path(entry).expanduser()
    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
        return Path(entry["path"]).expanduser()
    return None


def validate_model(path: Path, spec: ModelSpec) -> str | None:
    """Return None for a usable model, otherwise a short rejection reason."""
    if not path.is_dir():
        return "directory does not exist"
    config_path = path / "config.json"
    if not config_path.is_file():
        return "config.json is missing"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"config.json is invalid: {exc}"
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not any(
        architecture in spec.architectures for architecture in architectures
    ):
        expected = ", ".join(spec.architectures)
        return f"architecture does not match {expected}"
    weight_files = tuple(path.glob("*.safetensors")) + tuple(path.glob("*.bin"))
    if not any(weight.is_file() and weight.stat().st_size > 0 for weight in weight_files):
        return "model weights are missing"
    if spec.requires_tokenizer:
        tokenizer_files = (
            path / "tokenizer.json",
            path / "tokenizer.model",
            path / "vocab.json",
        )
        if not (path / "tokenizer_config.json").is_file() or not any(
            item.is_file() for item in tokenizer_files
        ):
            return "tokenizer files are missing"
    return None


def _candidate_paths(
    spec: ModelSpec,
    model_dir: Path,
    registry_models: Mapping[str, Any],
    environment: Mapping[str, str],
) -> list[Path]:
    candidates: list[Path] = []
    override = environment.get(spec.override_environment)
    if override:
        candidates.append(Path(override).expanduser())
    registered = _registry_model_path(registry_models, spec.key)
    if registered is not None:
        candidates.append(registered)
    candidates.append(model_dir / spec.directory_name)
    candidates.append(SHARED_MODEL_DIR / spec.directory_name)
    candidates.append(CONTAINER_MODEL_DIR / spec.directory_name)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized not in seen:
            unique.append(candidate)
            seen.add(normalized)
    return unique


def find_model(
    spec: ModelSpec,
    *,
    model_dir: Path | None = None,
    registry_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = os.environ if environment is None else environment
    root = default_model_dir(values) if model_dir is None else model_dir.expanduser()
    registry = (
        default_registry_path(values) if registry_path is None else registry_path.expanduser()
    )
    models = _load_registry(registry)
    for candidate in _candidate_paths(spec, root, models, values):
        if validate_model(candidate, spec) is None:
            return candidate.resolve()
    return None


def resolve_default_model(
    method: str,
    *,
    registry_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve the install-time default for Draft or EAGLE."""
    spec = _spec_for_method(method)
    if spec is None:
        return None
    return find_model(spec, registry_path=registry_path, environment=environment)


def _write_registry(path: Path, resolved: Mapping[str, Path]) -> None:
    document = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "models": {
            spec.key: {
                "method": spec.method,
                "path": str(resolved[spec.key]),
                "repo_id": spec.repo_id,
                "revision": spec.revision,
            }
            for spec in DEFAULT_MODEL_SPECS
            if spec.key in resolved
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        json.dump(document, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _download_model(spec: ModelSpec, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelStoreError(
            "huggingface_hub is required to download default models; install it in the "
            "selected vLLM environment"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=destination,
    )


def bootstrap_default_models(
    *,
    model_dir: Path | None = None,
    registry_path: Path | None = None,
    download: bool = True,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    values = os.environ if environment is None else environment
    root = default_model_dir(values) if model_dir is None else model_dir.expanduser()
    registry = (
        default_registry_path(values) if registry_path is None else registry_path.expanduser()
    )
    resolved: dict[str, Path] = {}
    for spec in DEFAULT_MODEL_SPECS:
        model = find_model(
            spec,
            model_dir=root,
            registry_path=registry,
            environment=values,
        )
        if model is None:
            if not download:
                raise ModelStoreError(
                    f"no usable {spec.key} model found; expected {spec.repo_id} under {root}"
                )
            destination = root / spec.directory_name
            print(f"Downloading {spec.repo_id} to {destination}", flush=True)
            _download_model(spec, destination)
            reason = validate_model(destination, spec)
            if reason is not None:
                raise ModelStoreError(
                    f"downloaded {spec.repo_id} is unusable at {destination}: {reason}"
                )
            model = destination.resolve()
        else:
            print(f"Using {spec.key} model at {model}", flush=True)
        resolved[spec.key] = model
    _write_registry(registry, resolved)
    print(f"Wrote model registry to {registry}", flush=True)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec-models",
        description="Detect or download the default vSpec Draft and EAGLE models.",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing models from Hugging Face (enabled by default).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    options = build_parser().parse_args(argv)
    try:
        bootstrap_default_models(
            model_dir=options.model_dir,
            registry_path=options.registry,
            download=options.download,
        )
    except ModelStoreError as exc:
        raise SystemExit(f"vSpec model setup: {exc}") from exc


if __name__ == "__main__":
    main()
