"""CPU-only contract tests for the public paired-evaluation entry points."""

from __future__ import annotations

import contextlib
import io
import json
import os
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
    def test_common_vllm_flags_are_frozen_and_private_caches_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env_file = root / "common-env.json"
            common = {
                **paired.COMMON_VLLM_FLAGS,
                "HUST_VSPEC_CAPTURE_SAFE_STATIC": "1",
                "HUST_VSPEC_HOT_STATIC_SHAPES": "14,42,70,98,126",
            }
            dump(env_file, common)
            self.assertEqual(paired.read_env(env_file, dynamic=False), common)
            setup = {
                "common_env": common,
                "dynamic_env": {},
                "evaluation_token": "test-only-private-token",
                "paths": {
                    "native_backend_package": "/private/native-backend",
                    "native_evaluation_static_dir": "/private/evaluation-static",
                    "native_evaluation_dir": "/private/evaluation",
                    "dynamic_backend_package": "/private/dynamic-backend",
                    "dynamic_evaluation_dir": "/private/evaluation",
                },
            }
            with patch.dict(
                os.environ,
                {
                    "ASCEND_API_KEY": "must-not-be-inherited",
                    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
                },
                clear=True,
            ):
                for arm in ("native", "dynamic"):
                    run_dir = root / arm
                    env = paired._server_env(setup, arm, run_dir)
                    self.assertNotIn("ASCEND_API_KEY", env)
                    self.assertEqual(env["VLLM_ALLOW_INSECURE_SERIALIZATION"], "0")
                    for key, value in paired.COMMON_VLLM_FLAGS.items():
                        self.assertEqual(env[key], value)
                    for key, suffix in paired.CACHE_DIRS.items():
                        self.assertEqual(env[key], str(run_dir / "cache" / suffix))
                        self.assertTrue(Path(env[key]).is_dir())
            dump(env_file, {"VLLM_WORKER_MULTIPROC_METHOD": "fork"})
            with self.assertRaisesRegex(ValueError, "differs from the matched profile"):
                paired.read_env(env_file, dynamic=False)
            dump(env_file, {"VLLM_UNREVIEWED_FLAG": "1"})
            with self.assertRaisesRegex(ValueError, "cannot set plugin/runtime policy"):
                paired.read_env(env_file, dynamic=False)

    def test_dynamic_backend_flags_are_explicitly_allowlisted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dynamic-env.json"
            dump(path, {"VLLM_ASCEND_Q35_GDN_INPUT_FUSION": "on"})
            self.assertEqual(
                paired.read_env(path, dynamic=True),
                {"VLLM_ASCEND_Q35_GDN_INPUT_FUSION": "on"},
            )
            dump(path, {"VLLM_ASCEND_Q35_UNREVIEWED_FLAG": "on"})
            with self.assertRaisesRegex(ValueError, "unexpected name"):
                paired.read_env(path, dynamic=True)

    def test_tokenizer_settings_are_not_mistaken_for_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "env.json"
            dump(path, {"TOKENIZERS_PARALLELISM": "false"})
            self.assertEqual(
                paired.read_env(path, dynamic=False), {"TOKENIZERS_PARALLELISM": "false"}
            )
            dump(path, {"HUST_VSPEC_SHARED_TOKENIZER_PADDING": "1"})
            self.assertEqual(
                paired.read_env(path, dynamic=True),
                {"HUST_VSPEC_SHARED_TOKENIZER_PADDING": "1"},
            )
            for secret_name in ("HF_TOKEN", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY"):
                dump(path, {secret_name: "secret-value"})
                with self.assertRaisesRegex(ValueError, "credentials"):
                    paired.read_env(path, dynamic=True)

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
                "pid": 100 + rank,
                "arm": "native",
                "configured_speculative_capacity": 4,
                "vspec_module_loaded": False,
                "dtype": "torch.bfloat16",
                "quantization": None,
                "explicit_kv_bytes": 2_000_000_000,
                "prefix_cache": True,
                "compiler": {
                    "actual_enable_static_kernel": True,
                    "actual_enable_npugraph_ex": True,
                    "cudagraph_mode": "FULL",
                    "all_capture_keys": [14, 42],
                },
                "static": {
                    "static_install": {
                        "rank": rank,
                        "pid": 100 + rank,
                        "install_owner_pid": 100 if rank == 0 else None,
                        "installed_package_hashes": ["pkg"] if rank == 0 else [],
                        "private_install_path": "/private/static-install",
                        "process_isolation": True,
                        "actual_installed_device_binary_sha256": {"kernel.o": "hash"},
                        "actual_installed_binary_metadata_sha256": {"kernel.json": "hash"},
                        "offline_compiler_jobs": 8,
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
        static_before = paired._static_install_signature("native", response)
        self.assertEqual(set(static_before), {0, 1})
        changed_install = json.loads(json.dumps(response))
        for worker in changed_install["results"]:
            worker["static"]["static_install"]["actual_installed_device_binary_sha256"][
                "kernel.o"
            ] = "new-hash"
        paired._validate_runtime_metadata("native", changed_install)
        self.assertNotEqual(
            static_before, paired._static_install_signature("native", changed_install)
        )
        paired._validate_operator_proof(
            "native",
            {
                "results": [
                    {
                        "arm": "native",
                        "software_paths": {"pid": 100},
                        "static_runtime": {
                            "vspec_plugin_absent": True,
                            "static_install": {"rank": 0, "pid": 100},
                        },
                    },
                    {
                        "arm": "native",
                        "software_paths": {"pid": 101},
                        "static_runtime": {
                            "vspec_plugin_absent": True,
                            "static_install": {"rank": 1, "pid": 101},
                        },
                    },
                ]
            },
            response,
            response,
        )
        response["results"][1]["max_model_len"] = 65536
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            paired._validate_runtime_metadata("native", response)

    def test_dynamic_operator_proof_requires_identity_and_observed_dispatch(self):
        metadata = {
            "results": [
                {"pid": 100 + rank, "static": {"static": {"rank": rank, "pid": 100 + rank}}}
                for rank in (0, 1)
            ]
        }

        def proof_row(rank, active):
            stride = [
                {"actual_replay_counts_by_gamma_and_requests": {"4x1": 1} if active else {}}
            ] + [{"actual_replay_counts_by_gamma_and_requests": {}} for _ in range(29)]
            edge = [{"actual_replay_counts_by_gamma_and_requests": {}} for _ in range(30)]
            rgdr = [{"actual_same_kernel_folded_replay_counter_cells": [0]} for _ in range(30)]
            return {
                "arm": "dynamic",
                "pid": 100 + rank,
                "static_runtime": {"static": {"rank": rank, "pid": 100 + rank}},
                "native_stride_convolution": stride,
                "native_edge_convolution": edge,
                "native_gamma8_RGDR": rgdr,
            }

        proof = {"results": [proof_row(0, True), proof_row(1, True)]}
        paired._validate_operator_proof("dynamic", proof, metadata, metadata)
        proof["results"][1]["pid"] = 999
        with self.assertRaisesRegex(RuntimeError, "PID"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata)
        proof["results"][1] = proof_row(1, False)
        with self.assertRaisesRegex(RuntimeError, "No dynamic native operator dispatch"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata)

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
            with patch.dict(
                paired.os.environ,
                {
                    "OPENAI_API_KEY": "private",
                    "HF_TOKEN": "private",
                    "AWS_SECRET_ACCESS_KEY": "private",
                    "PYTORCH_NPU_ALLOC_CONF": "test-setting",
                    "UNRELATED_VARIABLE": "excluded",
                },
            ):
                server_env = paired._server_env(
                    paired._load_setup(work / "setup.json"), "native", root / "native-server"
                )
            for key in (
                "OPENAI_API_KEY",
                "HF_TOKEN",
                "AWS_SECRET_ACCESS_KEY",
                "UNRELATED_VARIABLE",
            ):
                self.assertNotIn(key, server_env)
            self.assertEqual(server_env["PYTORCH_NPU_ALLOC_CONF"], "test-setting")
            self.assertEqual(
                server_env["PYTHONPATH"].split(":"), [str(backend), str(static), str(evaluation)]
            )
            self.assertEqual(server_env["VLLM_PLUGINS"], "ascend,vspec_eval")

    def test_compare_rejects_unproven_native_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in ("native", "dynamic"):
                directory = root / arm
                output = directory / "solver"
                for case_id in ("one", "two"):
                    dump(output / "requests" / f"{case_id}.json", {"case_id": case_id})
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
                        "plan_inventory_sha256": "inventory",
                        "selection_sha256": "selection",
                        "common_config_sha256": "common",
                        "setup_sha256": "setup",
                        "physical_npus": [0, 1],
                        "worker_runtime_common": {"dtype": "torch.bfloat16"},
                        "client_exit_code": 0,
                        "transport_or_response_failures": 0,
                        "output_tokens_per_second": 100,
                        "native_worker": None,
                        "selected_tracks": None,
                        "frontier_full_split": None,
                        "solver_output": str(output),
                        "record_count": 2,
                        "record_hashes": {
                            path.name: paired.sha256(path)
                            for path in (output / "requests").glob("*.json")
                        },
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
                        "plan_inventory_sha256": "inventory",
                        "quality": {"overall": 0.5},
                    },
                )
            args = Namespace(
                native_run=str(root / "native"), dynamic_run=str(root / "dynamic"), output=None
            )
            with self.assertRaisesRegex(ValueError, "observer"):
                paired.compare(args)
            dump(root / "native" / "solver" / "requests" / "one.json", {"case_id": "changed"})
            with self.assertRaisesRegex(ValueError, "changed after generation"):
                paired.compare(args)
            dump(root / "native" / "solver" / "requests" / "one.json", {"case_id": "one"})
            native_receipt_path = root / "native" / "solver-receipt.json"
            native_receipt = paired.read_json(native_receipt_path)
            native_receipt["native_worker"] = "native_static_worker.NativeEvaluationWorker"
            dump(native_receipt_path, native_receipt)
            dynamic_receipt_path = root / "dynamic" / "solver-receipt.json"
            dynamic_receipt = paired.read_json(dynamic_receipt_path)
            dynamic_receipt["output_tokens_per_second"] = 120
            dump(dynamic_receipt_path, dynamic_receipt)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), self.assertRaises(SystemExit) as exit_state:
                paired.compare(args)
            self.assertEqual(exit_state.exception.code, 2)
            result = json.loads(capture.getvalue())
            self.assertFalse(result["formal_scope_pass"])
            self.assertFalse(result["performance_quality_scope_pass"])
            self.assertNotIn("acceptance_pass", result)


if __name__ == "__main__":
    unittest.main()
