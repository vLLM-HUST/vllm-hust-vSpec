from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(
    script: str, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
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
        check=check,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_manage_install_editable_and_enable() -> None:
    result = run_script("manage.sh", "install", "--editable", "--enable")

    assert "pip install --no-deps --editable" in result.stdout
    assert "vllm_hust_vspec.model_store" not in result.stdout
    assert "extension inspect org.vllm-hust.vspec" in result.stdout
    assert "extension validate org.vllm-hust.vspec" in result.stdout
    assert "extension check org.vllm-hust.vspec" in result.stdout
    assert "extension enable org.vllm-hust.vspec" in result.stdout


def test_manage_has_no_model_download_interface() -> None:
    for arguments in (
        ("install", "--editable", "--model-download"),
        ("install", "--editable", "--model-dir", "/models"),
        ("models",),
    ):
        result = run_script("manage.sh", *arguments, check=False)
        assert result.returncode == 2
        assert "model_store" not in result.stdout


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


def test_qwen35_eagle3_run_preset_selects_qwen35_config() -> None:
    environment = os.environ.copy()
    environment["PYTHON_BIN"] = "/bin/echo"
    result = subprocess.run(
        [str(ROOT / "run.sh"), "qwen35-eagle3", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert "configs/qwen35-35b-a3b-eagle3.toml" in result.stdout
    assert "configs/qwen3-8b-eagle3.toml" not in result.stdout


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
