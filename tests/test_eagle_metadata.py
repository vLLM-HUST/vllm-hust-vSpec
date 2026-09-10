from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from vllm_hust_vspec.backends.eagle_metadata import _patch_metadata_cache


@dataclass
class _Metadata:
    logits_indices: np.ndarray
    target_logits_indices: np.ndarray
    draft_token_ids: np.ndarray


def test_metadata_cache_supports_current_two_argument_signature() -> None:
    class ModelRunner:
        calls = 0

        def _calc_spec_decode_metadata(
            self,
            num_draft_tokens: np.ndarray,
            cu_num_scheduled_tokens: np.ndarray,
        ) -> _Metadata:
            self.calls += 1
            return _Metadata(
                logits_indices=np.arange(6),
                target_logits_indices=np.array([0, 2]),
                draft_token_ids=np.array([-1, -1]),
            )

    assert _patch_metadata_cache(ModelRunner)
    runner = ModelRunner()
    runner.use_async_scheduling = True
    runner.speculative_config = SimpleNamespace(method="eagle")
    runner.dcp_size = 1
    runner.max_num_reqs = 2
    runner.num_spec_tokens = 2
    runner.arange_np = np.arange(3)
    runner.input_ids = SimpleNamespace(gpu=np.arange(10))
    num_draft_tokens = np.array([2, 2], dtype=np.int32)
    cu_num_scheduled_tokens = np.array([3, 6], dtype=np.int32)

    first = runner._calc_spec_decode_metadata(
        num_draft_tokens,
        cu_num_scheduled_tokens,
    )
    runner.input_ids.gpu += 10
    second = runner._calc_spec_decode_metadata(
        num_draft_tokens,
        cu_num_scheduled_tokens,
    )

    assert runner.calls == 1
    np.testing.assert_array_equal(first.draft_token_ids, np.array([-1, -1]))
    np.testing.assert_array_equal(second.draft_token_ids, np.array([11, 13]))
