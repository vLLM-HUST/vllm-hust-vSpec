from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from vllm_hust_vspec import manifests

EXTENSION_ID = "org.vllm-hust.vspec"


def _json_object(output: str) -> dict[str, object]:
    start = output.find("{")
    if start < 0:
        raise AssertionError(f"command did not emit a JSON object: {output!r}")
    value, _ = json.JSONDecoder().raw_decode(output[start:])
    assert isinstance(value, dict)
    return value


def _manager_is_available() -> bool:
    return _manager_path() is not None


def _manager_path() -> str | None:
    adjacent = Path(sys.executable).with_name("vllm-hust-ext")
    if adjacent.is_file():
        return str(adjacent)
    return shutil.which("vllm-hust-ext")


def _compatible_vllm_is_available() -> bool:
    try:
        installed = Version(importlib.metadata.version("vllm"))
    except importlib.metadata.PackageNotFoundError:
        return False
    manifest_path = Path(str(resources.files(manifests).joinpath("vllm-hust-extension-v0.2.json")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supported = SpecifierSet(manifest["host"]["version_range"])
    return installed in supported


@pytest.mark.skipif(
    not (_manager_is_available() and _compatible_vllm_is_available()),
    reason="requires Extension Manager and the tested vLLM-HUST host",
)
def test_extension_manager_lifecycle(tmp_path: Path) -> None:
    manager = _manager_path()
    assert manager is not None
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        }
    )

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [manager, *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )

    inspected = _json_object(run("extension", "inspect", EXTENSION_ID).stdout)
    assert inspected["activation_ready"] is True
    assert inspected["enabled"] is False

    run("extension", "validate", EXTENSION_ID)
    checked = _json_object(run("extension", "check", EXTENSION_ID).stdout)
    assert "compatible" in checked["states"]
    planned = _json_object(run("extension", "plan", EXTENSION_ID).stdout)
    assert planned["provider"] == "vllm"
    assert planned["generated_config"]["environment"] == {"HUST_VSPEC_ENABLED": "1"}
    rendered = run("extension", "render", EXTENSION_ID).stdout
    assert '"name": "vllm-launch.json"' in rendered

    run("extension", "enable", EXTENSION_ID)
    status = _json_object(run("extension", "status", EXTENSION_ID).stdout)
    assert "compatible" in status["states"]
    assert "enabled" in status["states"]

    dry_run = _json_object(
        run(
            "run",
            "--dry-run",
            "--",
            sys.executable,
            "-c",
            "raise AssertionError('dry-run command executed')",
        ).stdout
    )
    assert dry_run["environment"]["HUST_VSPEC_ENABLED"] == "1"
    assert dry_run["environment"]["VLLMHUST_EXT_ENABLED_BUNDLES"] == EXTENSION_ID

    run(
        "run",
        "--",
        sys.executable,
        "-c",
        (
            "import os; "
            "assert os.environ['HUST_VSPEC_ENABLED'] == '1'; "
            f"assert os.environ['VLLMHUST_EXT_ENABLED_BUNDLES'] == "
            f"'{EXTENSION_ID}'"
        ),
    )

    run("extension", "disable", EXTENSION_ID)
    run("extension", "forget", EXTENSION_ID)
    state_file = tmp_path / "config" / "vllm-hust" / "extensions.json"
    if state_file.exists():
        assert EXTENSION_ID not in state_file.read_text(encoding="utf-8")
