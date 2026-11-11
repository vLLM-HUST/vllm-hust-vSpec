"""CPU-only checks for the public issue #18 reproduction recipe."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/issue18_frozen56/reproduce.py"


def _module():
    spec = spec_from_file_location("issue18_reproduce", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prompt_format_and_answer_extraction():
    module = _module()
    row = {"category": "math", "question": "2+2?", "options": ["3", "4"]}
    shot = {"question": "1+1?", "options": ["1", "2"], "cot_content": "A: 1+1=2"}
    prompt = module.make_prompt(row, [shot])
    assert prompt.endswith("Question: 2+2?\nOptions: A. 3\nB. 4\nAnswer: Let's think step by step.\n\n")
    assert "Answer: 1+1=2\n\n" in prompt
    assert module.extract_answer("The answer is (B)") == "B"
    assert module.extract_answer("no option") is None


def test_profiles_keep_matching_capacity_and_frozen_shape_contract():
    engine = json.loads((ROOT / "configs/v0.15.2-issue18-frozen56.json").read_text())
    env = json.loads((ROOT / "configs/v0.15.2-issue18-frozen56-env.json").read_text())
    assert engine["common"]["max_num_seqs"] == 16
    assert engine["common"]["max_model_len"] == 16384
    assert engine["common"]["kv_cache_memory_bytes"] == 15569256448
    assert engine["native"]["speculative_config"]["num_speculative_tokens"] == 4
    assert engine["vspec"]["speculative_config"]["num_speculative_tokens"] == 8
    assert env["native"]["VLLM_PLUGINS"] == "ascend"
    assert env["vspec"]["VLLM_PLUGINS"] == "ascend,vspec"
    assert env["vspec"]["HUST_VSPEC_HOT_STATIC_SHAPES"] == "70"
    assert env["vspec"]["HUST_VSPEC_EVEN_POLICY_PROFILE"] == "archived-20261103"


def test_reuse_compiler_cache_relocates_triton_manifest(tmp_path):
    module = _module()
    source = tmp_path / "source"
    triton = source / "cache/triton/kernel"
    triton.mkdir(parents=True)
    (triton / "kernel.cubin").write_bytes(b"compiled")
    (triton / "__grp__kernel.json").write_text(json.dumps({
        "child_paths": {"kernel.cubin": str(triton / "kernel.cubin")}
    }))
    static = source / "static_kernel_compile_outputs"
    static.mkdir()
    (static / "marker").write_text("compiled")
    output = tmp_path / "output"
    output.mkdir()

    module.reuse_compiler_cache(source, output)

    copied = output / "cache/triton/kernel"
    manifest = json.loads((copied / "__grp__kernel.json").read_text())
    assert manifest["child_paths"]["kernel.cubin"] == str(copied / "kernel.cubin")
    assert (output / "static_kernel_compile_outputs/marker").read_text() == "compiled"
    assert (triton / "kernel.cubin").read_bytes() == b"compiled"


def test_reuse_compiler_cache_rejects_overlapping_paths(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="separate directories"):
        module.reuse_compiler_cache(tmp_path, tmp_path / "child")


def test_seed_triton_kernels_copies_only_diagnostic_families(tmp_path):
    module = _module()
    source = tmp_path / "source"
    families = (
        "postprocess_mamba_fused_kernel",
        "batch_memcpy_kernel",
        "gdn_gid123_full_stable_refresh",
    )
    for name in (*families, "unrelated_kernel"):
        directory = source / "cache/triton" / name
        directory.mkdir(parents=True)
        (directory / f"{name}.npubin").write_bytes(b"compiled")
        (directory / f"__grp__{name}.json").write_text(json.dumps({
            "child_paths": {f"{name}.npubin": str(directory / f"{name}.npubin")}
        }))
    (source / "static_kernel_compile_outputs").mkdir()
    output = tmp_path / "output"
    output.mkdir()

    copied = module.seed_triton_kernels(source, output)

    assert set(copied) == set(families)
    assert not (output / "cache/triton/unrelated_kernel").exists()
    assert not (output / "static_kernel_compile_outputs").exists()
    for name in families:
        directory = output / "cache/triton" / name
        manifest = json.loads((directory / f"__grp__{name}.json").read_text())
        assert manifest["child_paths"][f"{name}.npubin"] == str(directory / f"{name}.npubin")


def test_static_compiler_path_preflight():
    module = _module()
    module.validate_static_compile_path(Path("/root/issue18/r1"))
    with pytest.raises(ValueError, match="short --output"):
        module.validate_static_compile_path(Path("/root") / ("long-private-root-" * 6) / "r1")
