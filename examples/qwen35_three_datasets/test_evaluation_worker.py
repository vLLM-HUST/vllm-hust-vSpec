"""CPU-only checks for the standalone read-only worker metadata contract."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EVALUATION_DIR = Path(__file__).resolve().parents[2] / "src/vllm_hust_vspec/evaluation"
sys.path.insert(0, str(EVALUATION_DIR))
import evaluation_worker_common as worker_common  # noqa: E402
from evaluation_worker_common import EvaluationMetadataMixin  # noqa: E402


class FakeWorker(EvaluationMetadataMixin):
    def __init__(self, capacity=4):
        self.vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(num_speculative_tokens=capacity),
            model_config=SimpleNamespace(
                dtype="torch.bfloat16",
                quantization=None,
                max_model_len=32768,
                hf_text_config=SimpleNamespace(model_type="qwen3_5"),
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=16, max_num_batched_tokens=4096),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=2_000_000_000,
                enable_prefix_caching=True,
                mamba_cache_mode="align",
            ),
        )
        self.model_runner = SimpleNamespace(use_async_spec_decode=True, use_dcp=False)

    def get_eval_compiler_state(self):
        return {"actual_enable_static_kernel": True}

    def get_eval_static_state(self):
        return {"static_install": {"rank": 0, "process_isolation": True}}

    def get_private_even_width_state(self):
        return {"observed": True}


class WorkerTests(unittest.TestCase):
    def test_native_metadata_does_not_import_vspec(self):
        before = set(sys.modules)
        with patch.object(worker_common, "sys", SimpleNamespace(modules={})):
            with patch.dict(os.environ, {"VSPEC_EVAL_ARM": "native"}):
                result = FakeWorker().get_eval_metadata()
        self.assertEqual(result["configured_speculative_capacity"], 4)
        self.assertFalse(result["vspec_module_loaded"])
        self.assertFalse(
            any(
                name == "vllm_hust_vspec" or name.startswith("vllm_hust_vspec.")
                for name in set(sys.modules) - before
            )
        )

    def test_dynamic_requires_plugin_and_capacity_eight(self):
        modules = {}
        with patch.object(worker_common, "sys", SimpleNamespace(modules=modules)):
            with patch.dict(os.environ, {"VSPEC_EVAL_ARM": "dynamic"}):
                with self.assertRaisesRegex(RuntimeError, "did not load"):
                    FakeWorker(8).get_eval_metadata()
                with patch.dict(modules, {"vllm_hust_vspec": SimpleNamespace()}):
                    result = FakeWorker(8).get_eval_metadata()
                    self.assertEqual(result["dynamic_activity"], {"observed": True})


if __name__ == "__main__":
    unittest.main()
