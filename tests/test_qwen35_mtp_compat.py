"""CPU-only checks for the standalone, opt-in Qwen3.5 MTP bridge."""

from __future__ import annotations

import importlib.util
import os
import re
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

COMPAT_DIR = Path(__file__).resolve().parents[1] / "src/vllm_hust_vspec/_vendor/qwen35_mtp_compat"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, COMPAT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_layer_index(monkeypatch):
    utils = types.ModuleType("vllm.model_executor.models.utils")

    def extract_layer_index(name, num_attn_module):
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        assert match is not None
        return int(match.group(1))

    utils.extract_layer_index = extract_layer_index
    monkeypatch.setitem(sys.modules, utils.__name__, utils)


def test_kv_guard_allows_only_distinct_target_and_single_mtp_head(monkeypatch):
    _fake_layer_index(monkeypatch)
    kv = _load("qwen35_mtp_kv_compat")
    target, draft = object(), object()
    expected = {"model.layers.0.attn": target, "mtp.layers.0.attn": draft}
    assert kv._expected_qwen35_mtp_pair(expected, 1)
    assert not kv._expected_qwen35_mtp_pair(
        {"model.layers.0.attn": target, "other.layers.0.attn": draft}, 1
    )
    assert not kv._expected_qwen35_mtp_pair(
        {"model.layers.0.attn": target, "mtp.layers.0.attn": target}, 1
    )
    assert not kv._expected_qwen35_mtp_pair({**expected, "other.layers.0.attn": object()}, 1)


def test_kv_install_preserves_original_guard_for_unexpected_pairs(monkeypatch):
    _fake_layer_index(monkeypatch)
    worker = types.ModuleType("vllm.v1.worker")
    worker_utils = types.ModuleType("vllm.v1.worker.utils")
    platform = types.ModuleType("vllm_ascend.platform")

    class NPUPlatform:
        @classmethod
        def check_runner_kv_caches_multi_layer(cls):
            raise NotImplementedError("generic duplicate-index guard")

    def original_bind(*args):
        NPUPlatform.check_runner_kv_caches_multi_layer()

    worker.utils = worker_utils
    worker_utils.bind_kv_cache = original_bind
    platform.NPUPlatform = NPUPlatform
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker)
    monkeypatch.setitem(sys.modules, worker_utils.__name__, worker_utils)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    kv = _load("qwen35_mtp_kv_compat")
    assert kv.install()
    assert not kv.install()
    assert worker_utils.bind_kv_cache._qwen35_mtp_kv_guard
    worker_utils.bind_kv_cache(
        {"model.layers.0.attn": object(), "mtp.layers.0.attn": object()}, {}, []
    )
    with pytest.raises(NotImplementedError, match="generic duplicate-index guard"):
        worker_utils.bind_kv_cache(
            {"model.layers.0.attn": object(), "other.layers.0.attn": object()}, {}, []
        )
    with pytest.raises(NotImplementedError, match="generic duplicate-index guard"):
        NPUPlatform.check_runner_kv_caches_multi_layer()

    from vllm_hust_vspec.backends.kv_cache import apply_multi_layer_kv_cache_patch

    assert not apply_multi_layer_kv_cache_patch(NPUPlatform)


def test_mamba_bridge_is_idempotent_and_vspec_independent(monkeypatch):
    gpu = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    worker = types.ModuleType("vllm.v1.worker")
    torch = types.ModuleType("torch")
    observed = {}

    class Tensor:
        pass

    class GPUModelRunner:
        def _get_mamba_state_copy_funcs(self):
            return ("copy",)

    def original_copy(caches, num_blocks, copies):
        observed["planes"] = list(caches)

    torch.Tensor = Tensor
    gpu.GPUModelRunner = GPUModelRunner
    gpu.copy_kv_cache_blocks_inplace = original_copy
    worker.gpu_model_runner = gpu
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker)
    monkeypatch.setitem(sys.modules, gpu.__name__, gpu)
    bridge = _load("qwen35_mtp_mamba_compat")
    assert bridge.apply_mamba_runtime_compatibility_patch()
    assert not bridge.apply_mamba_runtime_compatibility_patch()
    first, second = Tensor(), Tensor()
    gpu.copy_kv_cache_blocks_inplace([(first, [second])], 2, [])
    assert observed["planes"] == [first, second]
    assert GPUModelRunner._get_mamba_state_copy_funcs._vspec_mamba_group_api


def test_unflagged_sitecustomize_does_not_import_runtime_modules():
    env = os.environ.copy()
    env.pop("QWEN35_MTP_COMPAT", None)
    env["PYTHONPATH"] = str(COMPAT_DIR)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'qwen35_mtp_kv_compat' not in sys.modules; "
            "assert 'vllm_hust_vspec' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_sitecustomize_activation_calls_standalone_bridges(monkeypatch):
    calls = []
    kv = types.ModuleType("qwen35_mtp_kv_compat")
    mamba = types.ModuleType("qwen35_mtp_mamba_compat")
    kv.install = lambda: calls.append("kv")
    mamba.apply_mamba_runtime_compatibility_patch = lambda: calls.append("mamba")
    monkeypatch.setitem(sys.modules, kv.__name__, kv)
    monkeypatch.setitem(sys.modules, mamba.__name__, mamba)
    monkeypatch.setenv("QWEN35_MTP_COMPAT", "1")
    runpy.run_path(str(COMPAT_DIR / "sitecustomize.py"))
    assert calls == ["kv", "mamba"]
    for name in ("sitecustomize.py", "qwen35_mtp_kv_compat.py", "qwen35_mtp_mamba_compat.py"):
        assert "import vllm_hust_vspec" not in (COMPAT_DIR / name).read_text(encoding="utf-8")


def test_sitecustomize_fails_closed_if_bridge_installation_fails(monkeypatch):
    kv = types.ModuleType("qwen35_mtp_kv_compat")
    mamba = types.ModuleType("qwen35_mtp_mamba_compat")

    def fail_install():
        raise RuntimeError("incompatible runtime")

    kv.install = fail_install
    mamba.apply_mamba_runtime_compatibility_patch = lambda: True
    monkeypatch.setitem(sys.modules, kv.__name__, kv)
    monkeypatch.setitem(sys.modules, mamba.__name__, mamba)
    monkeypatch.setenv("QWEN35_MTP_COMPAT", "1")
    with pytest.raises(SystemExit, match="compatibility startup failed: incompatible runtime"):
        runpy.run_path(str(COMPAT_DIR / "sitecustomize.py"))
