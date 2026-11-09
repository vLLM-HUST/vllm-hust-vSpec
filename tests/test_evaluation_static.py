"""CPU-only checks for the standalone native static evaluation adapter."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "src/vllm_hust_vspec/evaluation_static"
COMMON = ASSETS.parent / "evaluation"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"test_eval_static_{name}", ASSETS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_guard_defers_only_during_capture():
    capture = _load("capture_safe_static")
    active = False

    class Graph:
        def __init__(self):
            self.compiled_shape = set()

        def run_eagerly_compile(self, shape):
            self.compiled_shape.add(shape)
            return shape

    original, statistics = capture.install(SimpleNamespace(AclGraph=Graph), lambda: active)
    graph = Graph()
    active = True
    assert graph.run_eagerly_compile(14) is None
    assert graph.compiled_shape == set()
    active = False
    assert graph.run_eagerly_compile(14) == 14
    assert graph.compiled_shape == {14}
    assert statistics == {
        "static_compilation_deferred_inside_capture": 1,
        "original_eager_compile_method_calls": 1,
    }
    assert original is not Graph.run_eagerly_compile


def test_hot_shapes_preserve_supplied_options_and_validate_contract():
    hot = _load("hot_static_options")
    options = {
        "options": {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
            "static_kernel_compile": True,
            "_vllm_aclnn_static_kernel_sym_range": [1, 14, 35, 70, 126, 208],
        }
    }
    original = deepcopy(options)
    selected = hot.select(options)
    assert selected["options"]["_vllm_aclnn_static_kernel_sym_range"] == [14, 70, 126]
    assert options == original

    seen = []
    compiler = SimpleNamespace(_process_kwargs_options=lambda _config, opts: seen.append(opts))
    hot.install(compiler)
    compiler._process_kwargs_options(None, options)
    assert seen == [selected]
    assert hot.CALLS == [{"original_options": original, "selected_options": selected}]
    assert options == original

    bad = deepcopy(options)
    bad["options"]["clone_input"] = True
    with pytest.raises(ValueError, match="switches"):
        hot.select(bad)


def test_static_install_snapshot_proves_coordinator_and_receiver(tmp_path):
    metadata = _load("static_install_metadata")
    install_dir = tmp_path / "static_kernel_install" / "sample_1234"
    install_dir.mkdir(parents=True)
    binary = install_dir / "kernel.o"
    info = install_dir / "kernel.json"
    binary.write_bytes(b"device binary")
    info.write_text('{"name":"kernel"}', encoding="utf-8")
    static = SimpleNamespace(
        _use_process_isolation=True,
        _current_install_path=install_dir,
        _owns_install_path=1234,
        _installed_run_pkgs={"package-hash"},
        _compute_opc_compile_jobs=lambda: "8",
    )
    coordinator = metadata.snapshot(static, tmp_path, 0, 1234)
    static._owns_install_path = None
    static._installed_run_pkgs = set()
    receiver = metadata.snapshot(static, tmp_path, 1, 5678)
    assert (
        coordinator["actual_installed_device_binary_sha256"]
        == receiver["actual_installed_device_binary_sha256"]
    )
    assert (
        coordinator["actual_installed_binary_metadata_sha256"]
        == receiver["actual_installed_binary_metadata_sha256"]
    )
    assert metadata.verify_pair([coordinator, receiver])
    info.unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        metadata.snapshot(static, tmp_path, 1, 5678)


def test_native_software_path_observer_accepts_absent_q35_only_attributes(monkeypatch):
    def module(name, package=False):
        result = ModuleType(name)
        if package:
            result.__path__ = []
        monkeypatch.setitem(sys.modules, name, result)
        return result

    bf16 = object()
    none = object()
    module("torch").bfloat16 = bf16
    module("vllm_ascend", package=True)
    module("vllm_ascend.ops", package=True)
    module("vllm_ascend.ops.fused_moe", package=True)
    fused_moe = module("vllm_ascend.ops.fused_moe.fused_moe")
    module("vllm_ascend.quantization", package=True)
    quant_type = module("vllm_ascend.quantization.quant_type")
    quant_type.QuantType = SimpleNamespace(NONE=none)

    class AscendMoERunner:
        def __init__(self, name):
            self.layer_name = name
            self.ascend_shared_experts = SimpleNamespace(
                multistream_overlap=False, weights_replicated=False
            )
            self.quant_type = none
            self.gate = SimpleNamespace(weight=SimpleNamespace(dtype=bf16))

    fused_moe.AscendMoERunner = AscendMoERunner
    target = [AscendMoERunner(str(index)) for index in range(40)]
    draft = [AscendMoERunner("draft")]
    model = lambda owners: SimpleNamespace(
        named_modules=lambda: iter((str(index), owner) for index, owner in enumerate(owners))
    )
    observer = _load("vspec_readonly_observer").ReadOnlyVSpecWorkerObserver()
    observer.model_runner = SimpleNamespace(
        model=model(target), drafter=SimpleNamespace(model=model(draft))
    )
    proof = observer.get_vspec_readonly_software_paths()
    assert len(proof["rows"]) == 41
    assert all(row["original_same_and_cross_stream_events_restored"] for row in proof["rows"])
    target[0]._q35_single_stream_events = True
    assert not observer.get_vspec_readonly_software_paths()["rows"][0][
        "original_same_and_cross_stream_events_restored"
    ]


def test_native_worker_loads_without_plugin_or_npu(tmp_path):
    script = """
import sys
from types import ModuleType, SimpleNamespace

def module(name, package=False):
    result = ModuleType(name)
    if package:
        result.__path__ = []
    sys.modules[name] = result
    return result

torch_npu = module('torch_npu')
torch_npu.npu = SimpleNamespace(is_current_stream_capturing=lambda: False)
npugraph_ex = module('npugraph_ex', package=True)
npugraph_ex.CompilerConfig = object
configs = module('npugraph_ex.configs', package=True)
compiler_config = module('npugraph_ex.configs.compiler_config')
compiler_config._process_kwargs_options = lambda config, kwargs: kwargs
configs.compiler_config = compiler_config
concrete = module('npugraph_ex._acl_concrete_graph', package=True)
acl_graph = module('npugraph_ex._acl_concrete_graph.acl_graph')
class AclGraph:
    def run_eagerly_compile(self, *args, **kwargs):
        return None
acl_graph.AclGraph = AclGraph
static_kernel = module('npugraph_ex._acl_concrete_graph.static_kernel')
static_kernel._use_process_isolation = True
concrete.acl_graph = acl_graph
concrete.static_kernel = static_kernel

from native_static_worker import NativeEvaluationWorker
assert NativeEvaluationWorker.__module__ == 'native_static_worker'
assert all(callable(getattr(NativeEvaluationWorker, name)) for name in (
    'get_eval_metadata', 'get_eval_compiler_state', 'get_eval_static_state',
    'get_eval_operator_proof'))
assert AclGraph.run_eagerly_compile._vspec_capture_safe_static
assert compiler_config._process_kwargs_options._vspec_hot_static_metadata
assert not any(name == 'vllm_hust_vspec' or name.startswith('vllm_hust_vspec.')
    for name in sys.modules)
"""
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join((str(ASSETS), str(COMMON))),
        HUST_VSPEC_PRIVATE_COMPILER_ROOT=str(tmp_path),
        HUST_VSPEC_CAPTURE_SAFE_STATIC="1",
        HUST_VSPEC_HOT_STATIC_SHAPES="14,42,70,98,126",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
