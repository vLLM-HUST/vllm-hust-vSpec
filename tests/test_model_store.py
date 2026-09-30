from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from vllm_hust_vspec.model_store import (
    DEFAULT_MODEL_SPECS,
    missing_model_message,
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


def test_registry_resolves_existing_models(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    registry = tmp_path / "config" / "models.json"
    draft = create_model(model_dir, "draft")
    eagle = create_model(model_dir, "eagle")
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    "draft": {"path": str(draft)},
                    "eagle": {"path": str(eagle)},
                },
            }
        ),
        encoding="utf-8",
    )

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


def test_container_model_mount_is_discovered(tmp_path: Path) -> None:
    container_root = tmp_path / "model"
    draft = create_model(container_root, "draft")

    with (
        mock.patch(
            "vllm_hust_vspec.model_store.SHARED_MODEL_DIR",
            tmp_path / "unavailable-shared",
        ),
        mock.patch(
            "vllm_hust_vspec.model_store.CONTAINER_MODEL_DIR",
            container_root,
        ),
    ):
        resolved = resolve_default_model(
            "draft_model",
            registry_path=tmp_path / "missing-registry.json",
            environment={"HUST_VSPEC_MODEL_DIR": str(tmp_path / "other-models")},
        )

    assert resolved == draft.resolve()


def test_missing_model_message_is_actionable_and_has_no_download_command() -> None:
    message = missing_model_message("eagle")

    assert "Zjcxy-SmartAI/Eagle-Qwen2.5-14B-Instruct" in message
    assert "--draft-model PATH" in message
    assert "/model/Eagle-Qwen2.5-14B-Instruct" in message
    assert "does not download models" in message
    assert "manage.sh models" not in message


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
