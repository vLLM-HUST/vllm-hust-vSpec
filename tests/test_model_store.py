from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from vllm_hust_vspec.model_store import (
    DEFAULT_MODEL_SPECS,
    ModelStoreError,
    bootstrap_default_models,
    resolve_default_model,
    validate_model,
)


def create_model(root: Path, key: str) -> Path:
    spec = next(item for item in DEFAULT_MODEL_SPECS if item.key == key)
    path = root / spec.directory_name
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": [spec.architectures[0]]}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")
    if spec.requires_tokenizer:
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_bootstrap_detects_models_and_writes_registry(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    registry = tmp_path / "config" / "models.json"
    draft = create_model(model_dir, "draft")
    eagle = create_model(model_dir, "eagle")

    resolved = bootstrap_default_models(
        model_dir=model_dir,
        registry_path=registry,
        download=False,
        environment={},
    )

    assert resolved == {"draft": draft.resolve(), "eagle": eagle.resolve()}
    document = json.loads(registry.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["models"]["draft"]["path"] == str(draft.resolve())
    assert document["models"]["eagle"]["path"] == str(eagle.resolve())
    assert (
        resolve_default_model(
            "draft_model",
            registry_path=registry,
            environment={"HUST_VSPEC_MODEL_DIR": str(tmp_path / "elsewhere")},
        )
        == draft.resolve()
    )
    assert (
        resolve_default_model(
            "eagle",
            registry_path=registry,
            environment={"HUST_VSPEC_MODEL_DIR": str(tmp_path / "elsewhere")},
        )
        == eagle.resolve()
    )


def test_explicit_environment_model_precedes_registry(tmp_path: Path) -> None:
    registered_root = tmp_path / "registered"
    override_root = tmp_path / "override"
    registered = create_model(registered_root, "draft")
    override = create_model(override_root, "draft")
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {"draft": {"path": str(registered)}},
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_default_model(
        "draft_model",
        registry_path=registry,
        environment={"HUST_VSPEC_DRAFT_MODEL": str(override)},
    )

    assert resolved == override.resolve()


def test_bootstrap_downloads_only_missing_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    registry = tmp_path / "models.json"
    create_model(model_dir, "draft")

    def fake_download(spec, destination: Path) -> None:
        create_model(model_dir, spec.key)

    with (
        mock.patch(
            "vllm_hust_vspec.model_store._download_model",
            side_effect=fake_download,
        ) as call,
        mock.patch(
            "vllm_hust_vspec.model_store.SHARED_MODEL_DIR",
            tmp_path / "unavailable-shared",
        ),
    ):
        resolved = bootstrap_default_models(
            model_dir=model_dir,
            registry_path=registry,
            environment={},
        )

    assert set(resolved) == {"draft", "eagle"}
    assert call.call_count == 1
    assert call.call_args.args[0].key == "eagle"


def test_bootstrap_offline_rejects_missing_model(tmp_path: Path) -> None:
    with mock.patch(
        "vllm_hust_vspec.model_store.SHARED_MODEL_DIR",
        tmp_path / "unavailable-shared",
    ):
        with pytest.raises(ModelStoreError, match="no usable draft model"):
            bootstrap_default_models(
                model_dir=tmp_path / "models",
                registry_path=tmp_path / "models.json",
                download=False,
                environment={},
            )


def test_validation_rejects_wrong_architecture(tmp_path: Path) -> None:
    spec = next(item for item in DEFAULT_MODEL_SPECS if item.key == "eagle")
    path = tmp_path / "eagle"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (path / "model.safetensors").write_bytes(b"weights")

    assert validate_model(path, spec) == "architecture does not match Qwen2ForCausalLMEagle"
