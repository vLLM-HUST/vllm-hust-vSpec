"""CPU-only checks for paired live-width transition receipts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from vllm_hust_vspec.adaptive.even_fence_contract import consume


def _receipt(pid: int, roster: list[str], accepted: list[int], lengths: list[int]) -> dict:
    return {
        "eligible": True,
        "pid": pid,
        "transition_id": 1,
        "stage": "shrink",
        "old_gamma": 4,
        "new_gamma": 2,
        "old_actual_width": 5,
        "new_actual_width": 3,
        "roster": roster,
        "device_fenced": True,
        "cache_tensors_rewritten": False,
        "sampled_counts_rewritten": False,
        "previous_draft_tensor_rewritten": False,
        "previous_draft_width": 4,
        "previous_draft_stride": 4,
        "mamba_block_size": 2048,
        "accepted_counts": accepted,
        "old_end_seq_lens": lengths,
        "all30_cache_layouts_validated": True,
        "actual_width_cache_address_signature_validated": True,
        "groups": [
            {
                "gid": gid,
                "mamba_block_size": 2048,
                "previous_actual_canonical": True,
                "accepted_state_retained": True,
                "next_slots_allocated": True,
                "actual_previous_width": 5,
            }
            for gid in (1, 2, 3)
        ],
    }


def _admission(second: dict) -> tuple[SimpleNamespace, dict]:
    first = _receipt(11, ["request-a", "request-b"], [1, 2], [100, 150])
    runner = SimpleNamespace(
        _vspec_even_pending_fence=first,
        prev_num_spec_tokens=4,
        _draft_token_ids=SimpleNamespace(shape=(2, 4)),
    )
    marker = {
        "kind": "even_round",
        "transition_id": 1,
        "stage": "shrink",
        "old_gamma": 4,
        "new_gamma": 2,
        "queue_depth": 0,
        "placeholders_zero": True,
        "roster": ["request-a", "request-b"],
        "worker_receipts": [first, second],
    }
    return runner, marker


class EvenFenceContractTests(unittest.TestCase):
    def test_matching_checkpoints_in_different_roster_order_pass(self) -> None:
        second = _receipt(12, ["request-b", "request-a"], [2, 1], [150, 100])
        runner, marker = _admission(second)

        self.assertEqual(consume(runner, marker, 4, 2, 11), "shrink")
        self.assertIsNone(runner._vspec_even_pending_fence)
        self.assertEqual(runner._vspec_even_consumed_fence, (1, "shrink"))

    def test_matching_bridge_checkpoints_pass(self) -> None:
        second = _receipt(12, ["request-b", "request-a"], [2, 1], [150, 100])
        runner, marker = _admission(second)
        for receipt in marker["worker_receipts"]:
            receipt.update(stage="bridge", new_gamma=6, new_actual_width=7)
        marker.update(stage="bridge", new_gamma=6)

        self.assertEqual(consume(runner, marker, 4, 6, 11), "bridge")
        self.assertEqual(runner._vspec_even_consumed_fence, (1, "bridge"))

    def test_mismatched_accepted_count_fails_closed(self) -> None:
        second = _receipt(12, ["request-b", "request-a"], [1, 1], [150, 100])
        runner, marker = _admission(second)

        with self.assertRaisesRegex(RuntimeError, "TP workers disagree on per-request"):
            consume(runner, marker, 4, 2, 11)
        self.assertIs(runner._vspec_even_pending_fence, marker["worker_receipts"][0])

    def test_mismatched_sequence_length_fails_closed(self) -> None:
        second = _receipt(12, ["request-b", "request-a"], [2, 1], [150, 101])
        runner, marker = _admission(second)

        with self.assertRaisesRegex(RuntimeError, "TP workers disagree on per-request"):
            consume(runner, marker, 4, 2, 11)
        self.assertIs(runner._vspec_even_pending_fence, marker["worker_receipts"][0])


if __name__ == "__main__":
    unittest.main()
