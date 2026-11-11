"""Native evaluation checks the selected MTP width, not a hard-coded 4."""

from types import SimpleNamespace as NS

import pytest

from vllm_hust_vspec.evaluation import evaluation_worker_common as common


class Worker(common.EvaluationMetadataMixin):
    def __init__(self, width):
        self.vllm_config = NS(
            speculative_config=NS(num_speculative_tokens=width),
            model_config=NS(dtype="bf16", quantization=None, max_model_len=32768,
                            hf_text_config=NS(model_type="qwen3_5")),
            scheduler_config=NS(max_num_seqs=16, max_num_batched_tokens=8192),
            cache_config=NS(kv_cache_memory_bytes=None, enable_prefix_caching=True,
                            mamba_cache_mode="align"),
        )
        self.model_runner = NS(use_async_spec_decode=True, use_dcp=False)

    def get_eval_compiler_state(self):
        return {}

    def get_eval_static_state(self):
        return {}


@pytest.mark.parametrize("width", [4, 6, 8])
def test_native_capacity_matches_declared_arm(monkeypatch, width):
    monkeypatch.setenv("VSPEC_EVAL_ARM", "native")
    monkeypatch.setenv("VSPEC_EVAL_NATIVE_CAPACITY", str(width))
    monkeypatch.setattr(common, "sys", NS(modules={}))
    assert Worker(width).get_eval_metadata()["configured_speculative_capacity"] == width


def test_native_capacity_mismatch_is_rejected(monkeypatch):
    monkeypatch.setenv("VSPEC_EVAL_ARM", "native")
    monkeypatch.setenv("VSPEC_EVAL_NATIVE_CAPACITY", "6")
    monkeypatch.setattr(common, "sys", NS(modules={}))
    with pytest.raises(RuntimeError, match="not MTP6"):
        Worker(4).get_eval_metadata()
