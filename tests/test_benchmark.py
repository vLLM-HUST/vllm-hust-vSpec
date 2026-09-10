from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from benchmarks.offline_ab import configure_plugin, extract_gsm8k_answer


class GSM8KAnswerExtractionTest(unittest.TestCase):
    def test_hash_answer(self) -> None:
        self.assertEqual(extract_gsm8k_answer("work\n#### 1,234"), "1234")

    def test_boxed_answer(self) -> None:
        self.assertEqual(extract_gsm8k_answer(r"Therefore \\boxed{-2.5}."), "-2.5")

    def test_natural_language_answer(self) -> None:
        self.assertEqual(extract_gsm8k_answer("The final answer is: 42."), "42")

    def test_falls_back_to_last_number(self) -> None:
        self.assertEqual(extract_gsm8k_answer("First 3, then 7"), "7")


class BenchmarkModelSelectionTest(unittest.TestCase):
    def test_dflash_accepts_explicit_model_pair(self) -> None:
        args = Namespace(
            method="dflash",
            target_model=Path("/models/target"),
            draft_model=Path("/models/dflash"),
            manifest_family="qwen3_8b",
            adaptive_profile=None,
            baseline=True,
            batch_size=8,
            gamma=0,
            adaptive_min_gamma=1,
            adaptive_control_interval=1,
            adaptive_hysteresis=0.03,
            adaptive_min_observations=32,
            adaptive_max_gamma_step=1,
            adaptive_latency_calibration=True,
            adaptive_latency_ewma_weight=0.2,
            adaptive_gamma0_mode="sticky",
            adaptive_trace=False,
            adaptive_full_graph=False,
            adaptive_async=False,
            execution_mode="eager",
            component_stats_prefix=None,
            device=None,
        )
        method, target, draft, family = configure_plugin(args)
        self.assertEqual(method, "dflash")
        self.assertEqual(target, "/models/target")
        self.assertEqual(draft, "/models/dflash")
        self.assertEqual(family, "qwen3_8b")


if __name__ == "__main__":
    unittest.main()
