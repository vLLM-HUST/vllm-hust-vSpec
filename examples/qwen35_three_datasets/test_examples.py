"""CPU-only contract tests for the public paired-evaluation entry points."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import paired
import select_cases


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def common_config() -> dict:
    return {
        "dtype": "bfloat16",
        "tensor_parallel_size": 2,
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "enable_prefix_caching": True,
        "kv_cache_memory_bytes": 2_000_000_000,
        "compilation_config": {"level": 3},
        "speculative_config": {"method": "mtp", "draft_tensor_parallel_size": 2},
    }


class PublicExampleTests(unittest.TestCase):
    def test_configs_share_everything_except_depth_and_worker(self):
        pair = paired.aligned_configs(common_config(), "/models/qwen", "test-solver", 18080)
        self.assertEqual(pair["native"]["speculative_config"]["num_speculative_tokens"], 4)
        self.assertEqual(pair["dynamic"]["speculative_config"]["num_speculative_tokens"], 8)
        self.assertEqual(
            pair["native"]["worker_extension_cls"], "native_static_worker.NativeEvaluationWorker"
        )
        self.assertIn("DynamicEvaluationWorker", pair["dynamic"]["worker_extension_cls"])
        for arm in pair:
            pair[arm].pop("worker_extension_cls")
            pair[arm]["speculative_config"].pop("num_speculative_tokens")
        self.assertEqual(pair["native"], pair["dynamic"])

    def test_rejects_non_bf16_missing_kv_and_relative_private_output(self):
        config = common_config()
        config["dtype"] = "float16"
        with self.assertRaises(ValueError):
            paired.aligned_configs(config, "/model", "solver", 1)
        config = common_config()
        config["kv_cache_memory_bytes"] = 0
        with self.assertRaises(ValueError):
            paired.aligned_configs(config, "/model", "solver", 1)
        with self.assertRaises(ValueError):
            select_cases.private_output(Path("relative.json"))

    def test_selection_is_deterministic_and_plan_ordered(self):
        tasks = [
            {"case_id": f"case-{i}", "track": "olympiad" if i % 2 else "research"}
            for i in range(20)
        ]
        one = select_cases.choose_ids("frontierscience", tasks, 8, "public-seed")
        two = select_cases.choose_ids("frontierscience", tasks, 8, "public-seed")
        self.assertEqual(one, two)
        self.assertEqual(len(one), 8)
        self.assertEqual(sum(int(int(x.split("-")[1]) % 2 == 0) for x in one), 4)
        self.assertEqual(one, [row["case_id"] for row in tasks if row["case_id"] in one])

    def test_manifest_rejects_ids_absent_from_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            dump(plan / "contract.json", {"dataset": "hle-verified"})
            dump(plan / "tasks.json", [{"case_id": "real-1"}])
            ids = ["invented-1"]
            selection = root / "selection.json"
            dump(
                selection,
                {
                    "dataset": "hle-verified",
                    "source_plan_contract_sha256": paired.sha256(plan / "contract.json"),
                    "case_ids": ids,
                    "case_ids_sha256": paired.canonical_sha256(ids),
                    "count": 1,
                },
            )
            with self.assertRaisesRegex(ValueError, "absent"):
                paired._read_ids("hle-verified", plan, selection)

    def test_record_usage_rejects_incomplete_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            dump(
                output / "requests" / "one.json",
                {"case_id": "one", "status": "ok", "usage": {"completion_tokens": 10}},
            )
            self.assertEqual(
                paired._record_usage("hle-verified", output, ["one"])["completion_tokens"], 10
            )
            with self.assertRaises(RuntimeError):
                paired._record_usage("hle-verified", output, ["one", "two"])

    def test_runtime_and_operator_proof_cover_both_ranks(self):
        def row(rank):
            return {
                "arm": "native",
                "configured_speculative_capacity": 4,
                "vspec_module_loaded": False,
                "dtype": "torch.bfloat16",
                "quantization": None,
                "explicit_kv_bytes": 2_000_000_000,
                "prefix_cache": True,
                "compiler": {"actual_enable_static_kernel": True},
                "static": {
                    "static_install": {
                        "rank": rank,
                        "process_isolation": True,
                        "actual_installed_device_binary_sha256": {"kernel.o": "hash"},
                    }
                },
                "max_model_len": 32768,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 4096,
                "mamba_cache_mode": "align",
                "model_text_type": "qwen3_5",
            }

        response = {"results": [row(0), row(1)]}
        common = paired._validate_runtime_metadata("native", response)
        self.assertEqual(common["max_model_len"], 32768)
        paired._validate_operator_proof(
            "native",
            {
                "results": [
                    {
                        "arm": "native",
                        "static_runtime": {
                            "vspec_plugin_absent": True,
                            "static_install": {"rank": 0},
                        },
                    },
                    {
                        "arm": "native",
                        "static_runtime": {
                            "vspec_plugin_absent": True,
                            "static_install": {"rank": 1},
                        },
                    },
                ]
            },
        )
        response["results"][1]["max_model_len"] = 65536
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            paired._validate_runtime_metadata("native", response)

    def test_prepare_and_run_dry_run_without_npu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "env" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o700)
            vllm = python.parent / "vllm"
            vllm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            vllm.chmod(0o700)
            model = root / "model"
            backend = root / "backend"
            model.mkdir()
            backend.mkdir()
            (backend / "vllm_ascend").mkdir()
            (backend / "vllm_ascend" / "__init__.py").touch()
            evaluation = backend / "vllm_hust_vspec" / "evaluation"
            evaluation.mkdir(parents=True)
            for name in (
                "evaluation_endpoint.py",
                "evaluation_worker_common.py",
                "dynamic_worker.py",
            ):
                (evaluation / name).touch()
            static = backend / "vllm_hust_vspec" / "evaluation_static"
            static.mkdir()
            for name in (
                "native_static_worker.py",
                "capture_safe_static.py",
                "hot_static_options.py",
                "static_install_metadata.py",
                "vspec_readonly_observer.py",
                "vspec_readonly_compiler.py",
                "vspec_readonly_nge_worker.py",
                "vspec_readonly_static_worker.py",
            ):
                (static / name).touch()
            common = root / "common.json"
            common_env = root / "common-env.json"
            dynamic_env = root / "dynamic-env.json"
            dump(common, common_config())
            dump(
                common_env,
                {
                    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
                    "HUST_VSPEC_CAPTURE_SAFE_STATIC": "1",
                    "HUST_VSPEC_HOT_STATIC_SHAPES": "14,42,70,98,126",
                },
            )
            dump(dynamic_env, {})
            work = root / "private-run"
            setup_args = Namespace(
                private_root=str(work),
                model_dir=str(model),
                common_config=str(common),
                common_env=str(common_env),
                dynamic_env=str(dynamic_env),
                native_python=str(python),
                dynamic_python=str(python),
                client_python=str(python),
                native_backend_package=str(backend),
                dynamic_backend_package=str(backend),
                native_evaluation_dir=str(evaluation),
                dynamic_evaluation_dir=str(evaluation),
                served_model="test-solver",
                port=18080,
            )
            with patch.object(paired, "_active_evaluation_dir", return_value=evaluation):
                with contextlib.redirect_stdout(io.StringIO()):
                    paired.prepare(setup_args)
            plan = root / "mmlu-plan"
            dump(plan / "contract.json", {"dataset": "mmlu-pro"})
            dump(plan / "cases.json", [{"case_id": "one"}])
            run_args = Namespace(
                setup=str(work / "setup.json"),
                arm="native",
                dataset="mmlu-pro",
                plan=str(plan),
                selection=None,
                tag="dry-run",
                execute=False,
                startup_timeout=1,
                client_source=None,
            )
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                paired.run(run_args)
            command = json.loads(capture.getvalue())
            self.assertIn("--qwen35-mtp-compat", command["server"])
            self.assertEqual(command["case_count"], 1)
            self.assertFalse((work / "runs").exists())
            server_env = paired._server_env(
                paired._load_setup(work / "setup.json"), "native", root / "native-server"
            )
            self.assertEqual(
                server_env["PYTHONPATH"].split(":"), [str(backend), str(static), str(evaluation)]
            )
            self.assertEqual(server_env["VLLM_PLUGINS"], "ascend,vspec_eval")

    def test_compare_rejects_unproven_native_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in ("native", "dynamic"):
                directory = root / arm
                hashes = {}
                for name, key in (
                    ("worker-before.json", "worker_before_sha256"),
                    ("worker-after.json", "worker_after_sha256"),
                    ("cache-reset.json", "cache_reset_sha256"),
                    ("operator-proof.json", "operator_proof_sha256"),
                ):
                    dump(directory / name, {"arm": arm})
                    hashes[key] = paired.sha256(directory / name)
                dump(
                    directory / "solver-receipt.json",
                    {
                        "arm": arm,
                        "dataset": "mmlu-pro",
                        "case_count": 2,
                        "plan_contract_sha256": "plan",
                        "selection_sha256": "selection",
                        "common_config_sha256": "common",
                        "physical_npus": [0, 1],
                        "worker_runtime_common": {"dtype": "torch.bfloat16"},
                        "client_exit_code": 0,
                        "transport_or_response_failures": 0,
                        "output_tokens_per_second": 100,
                        "native_worker": None,
                        **hashes,
                    },
                )
                dump(
                    directory / "score-receipt.json",
                    {
                        "arm": arm,
                        "dataset": "mmlu-pro",
                        "selection_sha256": "selection",
                        "plan_contract_sha256": "plan",
                        "quality": {"overall": 0.5},
                    },
                )
            args = Namespace(
                native_run=str(root / "native"), dynamic_run=str(root / "dynamic"), output=None
            )
            with self.assertRaisesRegex(ValueError, "observer"):
                paired.compare(args)


if __name__ == "__main__":
    unittest.main()
