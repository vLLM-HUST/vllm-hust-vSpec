"""CPU-only contract tests for the public paired-evaluation entry points."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
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


def warmup_manifest(dataset: str, plan: Path, case_ids: list[str]) -> dict:
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source_plan_contract_sha256": paired.sha256(plan / "contract.json"),
        "count": len(case_ids),
        "case_ids": case_ids,
        "case_ids_sha256": paired.canonical_sha256(case_ids),
    }


class PublicExampleTests(unittest.TestCase):
    def test_native_operator_preflight_does_not_claim_dynamic_counter_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            before = {"results": [{"pid": 10}, {"pid": 11}]}
            proof = {"results": [{"arm": "native"}]}
            with (
                patch.object(paired, "_eval_request", return_value=proof),
                patch.object(paired, "_validate_operator_proof") as validate,
            ):
                self.assertIsNone(
                    paired._preflight_operator_proof({}, "native", run_dir, before)
                )
            validate.assert_called_once_with("native", proof, before, before)
            self.assertFalse((run_dir / "operator-before.json").exists())

    def test_dynamic_operator_preflight_saves_counter_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            proof = {"results": [{"arm": "dynamic"}]}
            with (
                patch.object(paired, "_eval_request", return_value=proof),
                patch.object(
                    paired, "_static_install_signature", return_value={0: {"pid": 10}, 1: {"pid": 11}}
                ),
                patch.object(paired, "_dynamic_counter_inventory") as inventory,
            ):
                self.assertIs(
                    paired._preflight_operator_proof({}, "dynamic", run_dir, {}), proof
                )
            inventory.assert_called_once_with(proof, {0: 10, 1: 11})
            self.assertEqual(paired.read_json(run_dir / "operator-before.json"), proof)

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
            cann = root / "cann"
            cann_site = cann / "python" / "site-packages"
            cann_site.mkdir(parents=True)
            (cann_site / "acl.py").write_text("TEST_CANN_ACL = True\n", encoding="utf-8")
            opp = cann / "opp"
            (opp / "built-in" / "op_impl" / "ai_core" / "tbe").mkdir(parents=True)
            setup = {
                "common_env": common,
                "dynamic_env": {},
                "evaluation_token": "test-only-private-token",
                "paths": {
                    "native_python": sys.executable,
                    "dynamic_python": sys.executable,
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
                    "ASCEND_HOME_PATH": str(cann),
                    "ASCEND_OPP_PATH": str(opp),
                    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
                },
                clear=True,
            ), patch.object(paired.Path, "home", return_value=root):
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
                    if arm == "dynamic":
                        self.assertEqual(
                            env["PYTHONPATH"].split(os.pathsep),
                            [
                                "/private/evaluation",
                                str(cann_site),
                                str(opp / "built-in" / "op_impl" / "ai_core" / "tbe"),
                            ],
                        )
                        self.assertNotIn("/private/dynamic-backend", env["PYTHONPATH"])
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

    def test_auto_kv_and_native_depth_variants(self):
        config = common_config()
        config["kv_cache_memory_bytes"] = None
        for capacity in (4, 6, 8):
            pair = paired.aligned_configs(
                config, "/models/qwen", "test-solver", 18080, capacity
            )
            self.assertIsNone(pair["native"]["kv_cache_memory_bytes"])
            self.assertIsNone(pair["dynamic"]["kv_cache_memory_bytes"])
            self.assertEqual(
                pair["native"]["speculative_config"]["num_speculative_tokens"], capacity
            )
            self.assertEqual(pair["dynamic"]["speculative_config"]["num_speculative_tokens"], 8)
        with self.assertRaisesRegex(ValueError, "candidate capacity"):
            paired.aligned_configs(config, "/models/qwen", "test-solver", 18080, 10)

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

    def test_executable_preserves_venv_path_and_private_root_resolves_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            self.assertEqual(paired.existing_executable(str(python), "python"), python)
            with self.assertRaisesRegex(ValueError, "explicit absolute path"):
                paired.existing_executable("venv/bin/python", "python")
            nonexecutable = root / "nonexecutable"
            nonexecutable.write_text("test", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "executable file"):
                paired.existing_executable(str(nonexecutable), "python")
            source_link = root / "source-link"
            source_link.symlink_to(paired.REPO_ROOT, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside the source repository"):
                paired.private_root(str(source_link / "private-output"))

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

    def test_warmup_manifest_allows_plan_cases_outside_timed_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            dump(plan / "contract.json", {"dataset": "hle-verified"})
            dump(plan / "tasks.json", [{"case_id": "timed"}, {"case_id": "extra"}])
            selection = root / "warmup.json"
            dump(selection, warmup_manifest("hle-verified", plan, ["extra", "timed"]))
            ids, manifest_sha256 = paired._read_warmup_ids(
                "hle-verified", plan, selection, ["timed"]
            )
            self.assertEqual(ids, ["extra", "timed"])
            self.assertEqual(manifest_sha256, paired.sha256(selection))
            dump(selection, warmup_manifest("hle-verified", plan, ["unknown"]))
            with self.assertRaisesRegex(ValueError, "frozen client plan"):
                paired._read_warmup_ids("hle-verified", plan, selection, ["timed"])
            dump(selection, warmup_manifest("hle-verified", plan, ["timed", "timed"]))
            with self.assertRaisesRegex(ValueError, "integrity"):
                paired._read_warmup_ids("hle-verified", plan, selection, ["timed"])

    def test_mmlu_warmup_requires_first_cases_and_uses_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            dump(plan / "contract.json", {"dataset": "mmlu-pro"})
            selection = root / "warmup.json"
            dump(selection, warmup_manifest("mmlu-pro", plan, ["one", "two"]))
            ids, _ = paired._read_warmup_ids("mmlu-pro", plan, selection, ["one", "two", "three"])
            command = paired._client_command(
                {
                    "paths": {"client_python": "/private/bin/python"},
                    "port": 18080,
                    "served_model": "solver",
                },
                "mmlu-pro",
                plan,
                root / "warmup",
                ids,
                warmup=True,
            )
            self.assertEqual(command[-2:], ["--limit", "2"])
            self.assertNotIn("--resume", command)
            dump(selection, warmup_manifest("mmlu-pro", plan, ["two"]))
            with self.assertRaisesRegex(ValueError, "first N"):
                paired._read_warmup_ids("mmlu-pro", plan, selection, ["one", "two", "three"])

    def test_warmup_fails_closed_before_reset_and_uses_separate_output(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            events = []

            def warmup(*args):
                events.append("warmup")
                return {"case_count": 1}

            def request(_setup, route, *, post=False):
                events.append("reset" if post else "metadata")
                return (
                    {
                        "prefix_cache_cleared": True,
                        "encoder_cache_cleared": True,
                        "mm_cache_cleared": True,
                    }
                    if post
                    else {"results": []}
                )

            with (
                patch.object(paired, "_run_untimed_warmup", side_effect=warmup),
                patch.object(paired, "_eval_request", side_effect=request),
                patch.object(paired, "_validate_runtime_metadata", return_value={}),
            ):
                paired._prepare_timing(
                    {}, "native", "hle-verified", run_dir, ["client"], ["one"], None
                )
            self.assertEqual(events, ["warmup", "metadata", "reset"])
            with (
                patch.object(
                    paired, "_run_untimed_warmup", side_effect=RuntimeError("warmup failed")
                ),
                patch.object(paired, "_eval_request") as request_mock,
            ):
                with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                    paired._prepare_timing(
                        {}, "native", "hle-verified", run_dir, ["client"], ["one"], None
                    )
                request_mock.assert_not_called()
            reset_dir = run_dir / "failed-reset"
            reset_dir.mkdir()
            with (
                patch.object(paired, "_run_untimed_warmup", return_value={"case_count": 1}),
                patch.object(
                    paired,
                    "_eval_request",
                    side_effect=[{"results": []}, {"prefix_cache_cleared": True}],
                ),
                patch.object(paired, "_validate_runtime_metadata", return_value={}),
            ):
                with self.assertRaisesRegex(RuntimeError, "cache reset"):
                    paired._prepare_timing(
                        {}, "native", "hle-verified", reset_dir, ["client"], ["one"], None
                    )

            private_run = run_dir / "isolated"
            private_run.mkdir()

            def failed_client(*args, **kwargs):
                dump(
                    private_run / "warmup" / "requests" / "one.json",
                    {"case_id": "one", "status": "error"},
                )
                return SimpleNamespace(returncode=0)

            with patch.object(paired.subprocess, "run", side_effect=failed_client):
                with self.assertRaisesRegex(RuntimeError, "Warmup failed"):
                    paired._run_untimed_warmup(
                        "hle-verified", private_run, ["client"], ["one"], None
                    )
            self.assertFalse((private_run / "solver").exists())
            with patch.object(paired.subprocess, "run") as client_mock:
                with self.assertRaises(FileExistsError):
                    paired._run_untimed_warmup(
                        "hle-verified", private_run, ["client"], ["one"], None
                    )
                client_mock.assert_not_called()

    def test_cache_reset_retries_idle_conflict_only(self):
        conflict = paired.urllib.error.HTTPError(
            "http://127.0.0.1/reset", 409, "busy", None, io.BytesIO(b"busy")
        )
        acknowledged = {
            "prefix_cache_cleared": True,
            "encoder_cache_cleared": True,
            "mm_cache_cleared": True,
        }
        with (
            patch.object(paired, "_eval_request", side_effect=[conflict, acknowledged]) as request,
            patch.object(paired.time, "sleep") as sleep,
        ):
            result = paired._reset_caches({})
        self.assertEqual(
            result, {**acknowledged, "idle_retry_attempts": [{"status": 409, "body": "busy"}]}
        )
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(0.5)

        server_error = paired.urllib.error.HTTPError(
            "http://127.0.0.1/reset", 500, "error", None, io.BytesIO(b"error")
        )
        with (
            patch.object(paired, "_eval_request", side_effect=server_error),
            patch.object(paired.time, "sleep") as sleep,
        ):
            with self.assertRaises(paired.urllib.error.HTTPError):
                paired._reset_caches({})
            sleep.assert_not_called()

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
            dump(output / "requests" / "one.json", {"case_id": "two", "status": "ok"})
            with self.assertRaisesRegex(RuntimeError, "filename"):
                paired._record_usage("hle-verified", output, ["two"])
            dump(
                output / "requests" / "one.json",
                {"case_id": "one", "status": "ok", "usage": {"completion_tokens": 10}},
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
        auto_response = json.loads(json.dumps(response))
        for worker in auto_response["results"]:
            worker["configured_speculative_capacity"] = 6
            worker["explicit_kv_bytes"] = None
        paired._validate_runtime_metadata("native", auto_response, 6, None)
        with self.assertRaisesRegex(RuntimeError, "KV allocation policy"):
            paired._validate_runtime_metadata("native", auto_response, 6, 2_000_000_000)
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

    def test_dynamic_operator_proof_requires_timed_counter_increase(self):
        metadata = {
            "results": [
                {"pid": 100 + rank, "static": {"static": {"rank": rank, "pid": 100 + rank}}}
                for rank in (0, 1)
            ]
        }

        def proof_row(rank, count):
            stride = [
                {"layer": "layer-0", "actual_replay_counts_by_gamma_and_requests": {"4x1": count}}
            ] + [
                {"layer": f"layer-{i}", "actual_replay_counts_by_gamma_and_requests": {}}
                for i in range(1, 30)
            ]
            edge = [
                {"layer": f"layer-{i}", "actual_replay_counts_by_gamma_and_requests": {}}
                for i in range(30)
            ]
            rgdr = [
                {"layer": f"layer-{i}", "actual_same_kernel_folded_replay_counter_cells": [0]}
                for i in range(30)
            ]
            return {
                "arm": "dynamic",
                "pid": 100 + rank,
                "static_runtime": {"static": {"rank": rank, "pid": 100 + rank}},
                "native_stride_convolution": stride,
                "native_edge_convolution": edge,
                "native_gamma8_RGDR": rgdr,
            }

        baseline = {"results": [proof_row(0, 1), proof_row(1, 1)]}
        proof = {"results": [proof_row(0, 2), proof_row(1, 2)]}
        deltas = paired._validate_operator_proof("dynamic", proof, metadata, metadata, baseline)
        self.assertEqual(deltas[0]["native_stride_convolution"], 1)
        self.assertEqual(deltas[1]["native_stride_convolution"], 1)
        proof["results"][1]["pid"] = 999
        with self.assertRaisesRegex(RuntimeError, "PID"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata, baseline)
        proof["results"][1] = proof_row(1, 1)
        with self.assertRaisesRegex(RuntimeError, "No timed dynamic native operator dispatch"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata, baseline)
        proof["results"][1] = proof_row(1, 0)
        with self.assertRaisesRegex(RuntimeError, "counter decreased"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata, baseline)
        with self.assertRaisesRegex(RuntimeError, "pre-timing counter snapshot"):
            paired._validate_operator_proof("dynamic", proof, metadata, metadata)

    def test_judged_quality_is_derived_from_complete_selected_counts(self):
        hle_solver = {"case_count": 2, "plan_contract_sha256": "plan"}
        hle_summary = {
            "contract_sha256": "plan",
            "scoring_policy": "strict_json",
            "tracks": {"gold": {"total": 2, "successes": 1, "unresolved": 1}},
        }
        hle_quality = paired._quality_from_summary("hle-verified", hle_summary, hle_solver)
        self.assertEqual(hle_quality["observed_lower_bound"], 0.5)
        self.assertEqual(hle_quality["unresolved"], 1)
        hle_summary["tracks"]["gold"]["total"] = 1
        with self.assertRaisesRegex(RuntimeError, "complete solver selection"):
            paired._quality_from_summary("hle-verified", hle_summary, hle_solver)

        frontier_solver = {
            "case_count": 3,
            "plan_contract_sha256": "plan",
            "selected_tracks": {"olympiad": 2, "research": 1},
        }
        frontier_summary = {
            "contract_sha256": "plan",
            "scoring_policy": "strict_final",
            "tracks": {
                "olympiad": {"total": 2, "successes": 1, "judge_unresolved": 1},
                "research": {"total": 1, "successes": 0, "judge_unresolved": 0},
            },
        }
        frontier_quality = paired._quality_from_summary(
            "frontierscience", frontier_summary, frontier_solver
        )
        self.assertEqual(frontier_quality["unresolved"], 1)
        frontier_summary["tracks"]["research"]["total"] = 0
        with self.assertRaisesRegex(RuntimeError, "selected track"):
            paired._quality_from_summary("frontierscience", frontier_summary, frontier_solver)

    def test_prepare_and_run_dry_run_without_npu(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            python = root / "env" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
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
            setup = paired.read_json(work / "setup.json")
            self.assertEqual(setup["paths"]["native_python"], str(python))
            self.assertEqual(setup["paths"]["dynamic_python"], str(python))
            self.assertEqual(setup["paths"]["client_python"], str(python))
            plan = root / "mmlu-plan"
            dump(plan / "contract.json", {"dataset": "mmlu-pro"})
            dump(plan / "cases.json", [{"case_id": "one"}])
            run_args = Namespace(
                setup=str(work / "setup.json"),
                arm="native",
                dataset="mmlu-pro",
                plan=str(plan),
                selection=None,
                warmup_selection=None,
                tag="dry-run",
                execute=False,
                startup_timeout=1,
                client_source=None,
            )
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                paired.run(run_args)
            command = json.loads(capture.getvalue())
            self.assertEqual(command["server"][0], str(python))
            self.assertIn(str(vllm), command["server"])
            self.assertIn("--qwen35-mtp-compat", command["server"])
            self.assertIsNone(command["warmup"])
            self.assertEqual(command["warmup_case_count"], 0)
            self.assertEqual(command["case_count"], 1)
            self.assertFalse((work / "runs").exists())
            warmup = root / "warmup.json"
            dump(warmup, warmup_manifest("mmlu-pro", plan, ["one"]))
            run_args.warmup_selection = str(warmup)
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                paired.run(run_args)
            warmup_command = json.loads(capture.getvalue())
            self.assertEqual(warmup_command["warmup_case_count"], 1)
            self.assertEqual(warmup_command["warmup"][-2:], ["--limit", "1"])
            self.assertIn(
                str(work / "runs" / "mmlu-pro" / "native" / "dry-run" / "warmup"),
                warmup_command["warmup"],
            )
            self.assertFalse((work / "runs").exists())
            cann = root / "cann"
            cann_site = cann / "python" / "site-packages"
            cann_site.mkdir(parents=True)
            (cann_site / "acl.py").write_text("TEST_CANN_ACL = True\n", encoding="utf-8")
            opp = cann / "opp"
            opp_tbe = opp / "built-in" / "op_impl" / "ai_core" / "tbe"
            opp_tbe.mkdir(parents=True)
            with patch.dict(
                paired.os.environ,
                {
                    "OPENAI_API_KEY": "private",
                    "HF_TOKEN": "private",
                    "AWS_SECRET_ACCESS_KEY": "private",
                    "PYTORCH_NPU_ALLOC_CONF": "test-setting",
                    "UNRELATED_VARIABLE": "excluded",
                    "PYTHONPATH": "/sensitive/inherited/path",
                    "ASCEND_HOME_PATH": str(cann),
                    "ASCEND_OPP_PATH": str(opp),
                },
                clear=True,
            ), patch.object(paired.Path, "home", return_value=root):
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
                server_env["PYTHONPATH"].split(":"),
                [str(backend), str(static), str(evaluation), str(cann_site), str(opp_tbe)],
            )
            self.assertEqual(server_env["VLLM_PLUGINS"], "ascend,vspec_eval")
            self.assertNotIn("/sensitive/inherited/path", server_env["PYTHONPATH"])
            probe = subprocess.run(
                [str(python), "-B", "-c", "import acl; print(acl.__file__)"],
                env=server_env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(Path(probe.stdout.strip()).resolve(), cann_site / "acl.py")

    def test_long_run_path_uses_short_private_collision_safe_ipc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            home.mkdir(mode=0o700)
            run_dir = root / ("long-run-root-" * 5) / "runs" / "hle-verified" / "native" / "final"
            run_dir.mkdir(parents=True)
            self.assertGreater(len(os.fsencode(str(run_dir / "ipc" / ("0" * 36)))), 107)
            cann = root / "cann"
            cann_site = cann / "python" / "site-packages"
            cann_site.mkdir(parents=True)
            (cann_site / "acl.py").write_text("TEST_CANN_ACL = True\n", encoding="utf-8")
            opp = cann / "opp"
            (opp / "built-in" / "op_impl" / "ai_core" / "tbe").mkdir(parents=True)
            setup = {
                "common_env": {
                    "ASCEND_RT_VISIBLE_DEVICES": "0,1",
                    "HUST_VSPEC_CAPTURE_SAFE_STATIC": "1",
                    "HUST_VSPEC_HOT_STATIC_SHAPES": "14,42,70,98,126",
                },
                "dynamic_env": {},
                "evaluation_token": "private-token",
                "paths": {
                    "native_python": sys.executable,
                    "native_backend_package": str(root),
                    "native_evaluation_static_dir": str(root),
                    "native_evaluation_dir": str(root),
                },
            }
            with patch.object(paired.Path, "home", return_value=home), patch.dict(
                os.environ,
                {"ASCEND_HOME_PATH": str(cann), "ASCEND_OPP_PATH": str(opp)},
                clear=True,
            ):
                first = paired._server_env(setup, "native", run_dir)
                second = paired._server_env(setup, "native", run_dir)
            first_ipc = Path(first["VLLM_RPC_BASE_PATH"])
            second_ipc = Path(second["VLLM_RPC_BASE_PATH"])
            self.assertNotEqual(first_ipc, second_ipc)
            self.assertEqual(first_ipc.parent, home / ".vspec-ipc")
            self.assertEqual(first["TMPDIR"], str(first_ipc))
            self.assertEqual(first["TMP"], str(first_ipc))
            self.assertEqual(first["TEMP"], str(first_ipc))
            self.assertEqual(first_ipc.stat().st_mode & 0o777, 0o700)
            self.assertLessEqual(
                len(os.fsencode(str(first_ipc / ("0" * 36)))),
                paired.ZMQ_IPC_PATH_MAX_BYTES,
            )

            long_home = root / ("h" * 70)
            long_home.mkdir(mode=0o700)
            with patch.object(paired.Path, "home", return_value=long_home):
                with self.assertRaisesRegex(RuntimeError, "too long"):
                    paired._private_ipc_dir()

    def test_compare_rejects_unproven_native_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for arm in ("native", "dynamic"):
                directory = root / arm
                output = directory / "solver"
                capture_keys = [5, 10] if arm == "native" else [1, 2, 3]
                install_key = "static_install" if arm == "native" else "static"
                for case_id in ("one", "two"):
                    dump(output / "requests" / f"{case_id}.json", {"case_id": case_id})
                hashes = {}
                for name, key in (
                    ("worker-before.json", "worker_before_sha256"),
                    ("worker-after.json", "worker_after_sha256"),
                    ("cache-reset.json", "cache_reset_sha256"),
                    ("operator-proof.json", "operator_proof_sha256"),
                ):
                    proof = {"arm": arm}
                    if name in ("worker-before.json", "worker-after.json", "operator-proof.json"):
                        is_operator = name == "operator-proof.json"
                        state_key = "static_runtime" if is_operator else "static"
                        proof["results"] = [
                            {
                                **(
                                    {}
                                    if is_operator
                                    else {"compiler": {"all_capture_keys": capture_keys}}
                                ),
                                state_key: {
                                    "compiler": {"all_capture_keys": capture_keys},
                                    install_key: {"rank": rank},
                                },
                            }
                            for rank in (0, 1)
                        ]
                    dump(directory / name, proof)
                    hashes[key] = paired.sha256(directory / name)
                if arm == "dynamic":
                    dump(directory / "operator-before.json", {"arm": arm, "phase": "before"})
                    hashes["operator_before_sha256"] = paired.sha256(
                        directory / "operator-before.json"
                    )
                summary_path = directory / "mmlu-summary.json"
                dump(
                    summary_path,
                    {"complete": True, "accuracy": {"cases": 2, "correct": 1, "accuracy": 0.5}},
                )
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
                        "worker_runtime_common": {
                            "dtype": "torch.bfloat16",
                            "compiler": {
                                "cudagraph_mode": "FULL",
                                "actual_enable_npugraph_ex": True,
                                "actual_enable_static_kernel": True,
                                "all_capture_keys": capture_keys,
                            },
                        },
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
                        "quality": {"overall": 0.5, "unresolved": 0},
                        "summary_sha256": paired.sha256(summary_path),
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
            self.assertIn("score_receipt_sha256", result["scoring_artifacts_sha256"]["native"])

            # Native MTP4 and dynamic speculation compile different capture shapes;
            # those keys need not match across arms, but must match each arm's proof.
            dynamic_proof_path = root / "dynamic" / "operator-proof.json"
            dynamic_proof = paired.read_json(dynamic_proof_path)
            dynamic_proof["results"][0]["static_runtime"]["compiler"]["all_capture_keys"] = [99]
            dump(dynamic_proof_path, dynamic_proof)
            dynamic_receipt["operator_proof_sha256"] = paired.sha256(dynamic_proof_path)
            dump(dynamic_receipt_path, dynamic_receipt)
            with self.assertRaisesRegex(ValueError, "captured compiler keys differ"):
                paired.compare(args)
            dynamic_proof["results"][0]["static_runtime"]["compiler"][
                "all_capture_keys"
            ] = [1, 2, 3]
            dump(dynamic_proof_path, dynamic_proof)
            dynamic_receipt["operator_proof_sha256"] = paired.sha256(dynamic_proof_path)
            dump(dynamic_receipt_path, dynamic_receipt)

            dynamic_receipt["worker_runtime_common"]["compiler"]["cudagraph_mode"] = "NONE"
            dump(dynamic_receipt_path, dynamic_receipt)
            with self.assertRaisesRegex(ValueError, "worker_runtime_common"):
                paired.compare(args)
            dynamic_receipt["worker_runtime_common"]["compiler"]["cudagraph_mode"] = "FULL"
            dump(dynamic_receipt_path, dynamic_receipt)

            native_summary = root / "native" / "mmlu-summary.json"
            dump(
                native_summary,
                {"complete": True, "accuracy": {"cases": 2, "correct": 0, "accuracy": 0.0}},
            )
            with self.assertRaisesRegex(ValueError, "altered native score summary"):
                paired.compare(args)
            dump(
                native_summary,
                {"complete": True, "accuracy": {"cases": 2, "correct": 1, "accuracy": 0.5}},
            )
            score_path = root / "native" / "score-receipt.json"
            score_result = paired.read_json(score_path)
            score_result["quality"]["overall"] = 0.0
            dump(score_path, score_result)
            with self.assertRaisesRegex(ValueError, "quality differs"):
                paired.compare(args)
            score_result["quality"]["overall"] = 0.5
            dump(score_path, score_result)

            for arm in ("native", "dynamic"):
                directory = root / arm
                warmup_output = directory / "warmup"
                warmup_record = warmup_output / "requests" / "one.json"
                dump(warmup_record, {"case_id": "one", "status": "ok"})
                warmup_receipt = {
                    "schema_version": 1,
                    "dataset": "mmlu-pro",
                    "case_count": 1,
                    "case_ids_sha256": paired.canonical_sha256(["one"]),
                    "client_exit_code": 0,
                    "warmup_output": str(warmup_output),
                    "untimed": True,
                    "transport_or_response_failures": 0,
                    "record_hashes": {"one.json": paired.sha256(warmup_record)},
                }
                dump(directory / "warmup-receipt.json", warmup_receipt)
                solver_path = directory / "solver-receipt.json"
                solver = paired.read_json(solver_path)
                solver.update(
                    warmup_manifest_sha256="same-manifest",
                    warmup_case_ids_sha256=warmup_receipt["case_ids_sha256"],
                    warmup_case_count=1,
                    warmup_receipt_sha256=paired.sha256(directory / "warmup-receipt.json"),
                    cache_reset_after_warmup=True,
                )
                dump(solver_path, solver)
            warmup_record = root / "native" / "warmup" / "requests" / "one.json"
            dump(warmup_record, {"case_id": "changed", "status": "ok"})
            with self.assertRaisesRegex(ValueError, "Warmup request records changed"):
                paired.compare(args)
            dump(warmup_record, {"case_id": "one", "status": "ok"})
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                paired.compare(args)
            dynamic_path = root / "dynamic" / "solver-receipt.json"
            dynamic = paired.read_json(dynamic_path)
            dynamic["warmup_manifest_sha256"] = "different-manifest"
            dump(dynamic_path, dynamic)
            with self.assertRaisesRegex(ValueError, "warmup_manifest_sha256"):
                paired.compare(args)


if __name__ == "__main__":
    unittest.main()
