from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "VSPEC_MANAGER_BIN": "/bin/echo",
            "VSPEC_MANAGE_DRY_RUN": "1",
        }
    )
    return subprocess.run(
        [str(ROOT / script), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_manage_install_editable_and_enable() -> None:
    result = run_script("manage.sh", "install", "--editable", "--enable")

    assert "pip install --no-deps --editable" in result.stdout
    assert "-m vllm_hust_vspec.model_store" in result.stdout
    assert "extension inspect org.vllm-hust.vspec" in result.stdout
    assert "extension validate org.vllm-hust.vspec" in result.stdout
    assert "extension check org.vllm-hust.vspec" in result.stdout
    assert "extension enable org.vllm-hust.vspec" in result.stdout


def test_manage_install_model_setup_controls() -> None:
    result = run_script(
        "manage.sh",
        "install",
        "--editable",
        "--model-dir",
        "/models",
        "--model-registry",
        "/config/models.json",
        "--no-model-download",
    )

    assert "--model-dir /models" in result.stdout
    assert "--registry /config/models.json" in result.stdout
    assert "--no-download" in result.stdout

    skipped = run_script("manage.sh", "install", "--editable", "--skip-model-setup")
    assert "vllm_hust_vspec.model_store" not in skipped.stdout


def test_manage_models_command() -> None:
    result = run_script("manage.sh", "models", "--no-download")

    assert "-m vllm_hust_vspec.model_store --no-download" in result.stdout


def test_manage_uninstall_cleans_intent_and_only_removes_vspec() -> None:
    result = run_script("manage.sh", "uninstall")

    assert "extension disable org.vllm-hust.vspec" in result.stdout
    assert "extension forget org.vllm-hust.vspec" in result.stdout
    assert "pip uninstall -y vllm-hust-vspec" in result.stdout
    assert "vllm-ascend" not in result.stdout
    assert "torch-npu" not in result.stdout


def test_shortcut_scripts_delegate_to_manager() -> None:
    install = run_script("install.sh", "--enable")
    uninstall = run_script("uninstall.sh")

    assert "pip install --no-deps --editable" in install.stdout
    assert "extension enable org.vllm-hust.vspec" in install.stdout
    assert "pip uninstall -y vllm-hust-vspec" in uninstall.stdout


def test_manage_exposes_admission_and_render_commands() -> None:
    for command in ("validate", "inspect", "check", "status", "plan", "render"):
        result = run_script("manage.sh", command)
        assert f"extension {command} org.vllm-hust.vspec" in result.stdout

    listed = run_script("manage.sh", "list", "--json")
    assert "extension list --json" in listed.stdout


def test_manage_wraps_manager_dry_run() -> None:
    result = run_script("manage.sh", "run", "--dry-run", "--", sys.executable, "-c", "print(1)")

    assert "run --dry-run --" in result.stdout
    assert "print\\(1\\)" in result.stdout


def test_manage_upgrade_is_exact_and_checked() -> None:
    result = run_script("manage.sh", "upgrade", "--version", "0.12.0", "--enable")

    assert "pip install --no-cache-dir --upgrade" in result.stdout
    assert "vllm-hust-vspec==0.12.0" in result.stdout
    assert "extension validate org.vllm-hust.vspec" in result.stdout
    assert "extension check org.vllm-hust.vspec" in result.stdout
    assert "extension enable org.vllm-hust.vspec" in result.stdout
