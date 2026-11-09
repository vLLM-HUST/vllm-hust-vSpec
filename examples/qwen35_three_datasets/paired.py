"""Private-output launcher and ledger for matched Qwen3.5 MTP4/vSpec evaluations.

Only Python's standard library is needed here. The vLLM stack, plugin, model,
proof artifacts, frozen plans and dataset clients are supplied by the operator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES = {
    "mmlu-pro": "hust_eval_mmlu_pro",
    "hle-verified": "hust_eval_hle_verified",
    "frontierscience": "hust_eval_frontierscience",
}
REQUIRED_COMMON = {
    "dtype",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "kv_cache_memory_bytes",
    "compilation_config",
    "speculative_config",
}
CACHE_DIRS = {
    "XDG_CACHE_HOME": "xdg",
    "XDG_CONFIG_HOME": "xdg-config",
    "XDG_DATA_HOME": "xdg-data",
    "XDG_RUNTIME_DIR": "xdg-runtime",
    "VLLM_CACHE_ROOT": "vllm",
    "VLLM_CONFIG_ROOT": "vllm-config",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "TORCH_HOME": "torch",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "HF_HOME": "huggingface",
    "HF_HUB_CACHE": "huggingface/hub",
    "HF_DATASETS_CACHE": "huggingface/datasets",
    "ASCEND_WORK_PATH": "ascend-work",
    "ASCEND_PROCESS_LOG_PATH": "ascend-logs",
    "ASCEND_CACHE_PATH": "ascend-cache",
    "ACL_OP_COMPILER_CACHE_DIR": "acl-op",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def private_root(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not Path(value).is_absolute() or path.is_relative_to(REPO_ROOT):
        raise ValueError(
            "Private work directory must be absolute and outside the source repository"
        )
    return path


def existing_path(value: str, kind: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute() or "<" in value:
        raise ValueError(f"{kind} must be an explicit absolute path, not a placeholder")
    path = Path(value).expanduser().resolve(strict=True)
    return path


def _active_evaluation_dir(python: str) -> Path:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    command = [
        python,
        "-c",
        "import importlib.util; spec=importlib.util.find_spec('vllm_hust_vspec'); "
        "print(spec.origin if spec is not None else '')",
    ]
    origin = subprocess.check_output(command, text=True, env=env, timeout=20).strip()
    if not origin or not Path(origin).is_absolute():
        raise RuntimeError("Selected Python does not have an installed vSpec package")
    return Path(origin).resolve(strict=True).parent / "evaluation"


def read_env(path: Path, *, dynamic: bool) -> dict[str, str]:
    values = read_json(path)
    if not isinstance(values, dict):
        raise ValueError("Environment file must be a JSON object")
    forbidden = ("token", "secret", "password", "api_key", "credential")
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value or "<" in value:
            raise ValueError("Environment contains a placeholder or non-string value")
        if any(word in key.lower() for word in forbidden) or key in {"VLLM_PLUGINS", "PYTHONPATH"}:
            raise ValueError(
                f"Do not put credentials or launcher-reserved keys in environment file: {key}"
            )
        if dynamic and not key.startswith(("VSPEC_", "HUST_VSPEC_", "HUST_ISSUE18_")):
            raise ValueError(f"Dynamic-only setting has an unexpected name: {key}")
        common_static_keys = {
            "HUST_VSPEC_CAPTURE_SAFE_STATIC",
            "HUST_VSPEC_HOT_STATIC_SHAPES",
        }
        if (
            not dynamic
            and key.startswith(("VSPEC_", "HUST_", "VLLM_"))
            and key not in common_static_keys
        ):
            raise ValueError(f"Common environment cannot set plugin/runtime policy: {key}")
        if dynamic and key.endswith(("_DIR", "_ROOT", "_PATH")):
            existing_path(value, f"{key} proof/dependency path")
    return values


def aligned_configs(
    common: dict[str, Any], model: str, alias: str, port: int
) -> dict[str, dict[str, Any]]:
    if REQUIRED_COMMON - set(common):
        raise ValueError(f"Shared server config is missing {sorted(REQUIRED_COMMON - set(common))}")
    if common["dtype"] != "bfloat16" or common.get("quantization") is not None:
        raise ValueError("This matched example requires pure BF16 without quantization")
    if common["tensor_parallel_size"] != 2:
        raise ValueError("This Qwen3.5 example requires tensor parallel size 2")
    if type(common["kv_cache_memory_bytes"]) is not int or common["kv_cache_memory_bytes"] <= 0:
        raise ValueError("Set an explicit positive KV-cache byte budget for both arms")
    if not isinstance(common["compilation_config"], dict):
        raise ValueError("compilation_config must be identical JSON for both arms")
    if (
        not isinstance(common["speculative_config"], dict)
        or common["speculative_config"].get("method") != "mtp"
    ):
        raise ValueError("Shared speculative method must be mtp")
    if "num_speculative_tokens" in common["speculative_config"] or "worker_extension_cls" in common:
        raise ValueError("Arm-specific depth/worker must not be present in shared config")
    for key in ("model", "host", "port", "served_model_name"):
        if key in common:
            raise ValueError(f"{key} is supplied by this launcher, not the shared config")
    pair: dict[str, dict[str, Any]] = {}
    for arm, capacity in (("native", 4), ("dynamic", 8)):
        cfg = copy.deepcopy(common)
        cfg.update(model=model, host="127.0.0.1", port=port, served_model_name=alias)
        cfg["speculative_config"]["num_speculative_tokens"] = capacity
        cfg["worker_extension_cls"] = (
            "native_static_worker.NativeEvaluationWorker"
            if arm == "native"
            else "dynamic_worker.DynamicEvaluationWorker"
        )
        pair[arm] = cfg
    normalized = copy.deepcopy(pair)
    for cfg in normalized.values():
        cfg.pop("worker_extension_cls", None)
        cfg["speculative_config"].pop("num_speculative_tokens")
    if normalized["native"] != normalized["dynamic"]:
        raise AssertionError("Shared server settings diverged")
    return pair


def prepare(args: argparse.Namespace) -> None:
    root = private_root(args.private_root)
    if root.exists():
        raise FileExistsError(f"Use a new private root, not an existing run: {root}")
    model = existing_path(args.model_dir, "model directory")
    common_file = existing_path(args.common_config, "shared server configuration")
    common = read_json(common_file)
    if not isinstance(common, dict):
        raise ValueError("Shared server configuration must be a JSON object")
    common_env_file = existing_path(args.common_env, "common environment")
    dynamic_env_file = existing_path(args.dynamic_env, "dynamic environment/proof map")
    common_env = read_env(common_env_file, dynamic=False)
    dynamic_env = read_env(dynamic_env_file, dynamic=True)
    if set(common_env) & set(dynamic_env):
        raise ValueError("Dynamic environment must not override a common setting")
    paths = {
        key: str(existing_path(getattr(args, key), key))
        for key in (
            "native_python",
            "dynamic_python",
            "client_python",
            "native_backend_package",
            "dynamic_backend_package",
            "native_evaluation_dir",
            "dynamic_evaluation_dir",
        )
    }
    for key in ("native_python", "dynamic_python", "client_python"):
        if not os.access(paths[key], os.X_OK):
            raise ValueError(f"{key} is not executable")
    for arm in ("native", "dynamic"):
        backend_init = Path(paths[f"{arm}_backend_package"]) / "vllm_ascend" / "__init__.py"
        if not backend_init.is_file():
            raise FileNotFoundError(f"Selected backend package missing: {backend_init}")
    if args.port < 1 or args.port > 65535 or not args.served_model:
        raise ValueError("A valid loopback port and served model name are required")
    pair = aligned_configs(common, str(model), args.served_model, args.port)
    for arm in ("native", "dynamic"):
        directory = Path(paths[f"{arm}_evaluation_dir"])
        if directory.name != "evaluation" or directory.parent.name != "vllm_hust_vspec":
            raise ValueError("Evaluation assets must come from an installed vSpec package")
        if directory != _active_evaluation_dir(paths[f"{arm}_python"]):
            raise ValueError(f"{arm} evaluation assets differ from the active Python package")
        for name in (
            "evaluation_endpoint.py",
            "evaluation_worker_common.py",
            "dynamic_worker.py",
        ):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Installed evaluation asset missing: {directory / name}")
    native_static_dir = Path(paths["native_evaluation_dir"]).parent / "evaluation_static"
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
        if not (native_static_dir / name).is_file():
            raise FileNotFoundError(
                f"Installed native static asset missing: {native_static_dir / name}"
            )
    paths["native_evaluation_static_dir"] = str(native_static_dir)
    root.mkdir(parents=True, mode=0o700)
    config_dir = root / "server-config"
    config_dir.mkdir(mode=0o700)
    for arm, cfg in pair.items():
        # JSON is valid YAML. vLLM's --config parser expects dashed top-level names.
        write_new(
            config_dir / f"{arm}.yaml", {key.replace("_", "-"): value for key, value in cfg.items()}
        )
    setup = {
        "schema_version": 1,
        "private_root": str(root),
        "model_dir": str(model),
        "model_small_files_only_not_weight_hash": True,
        "port": args.port,
        "served_model": args.served_model,
        "paths": paths,
        "common_env": common_env,
        "dynamic_env": dynamic_env,
        "common_config_sha256": canonical_sha256(common),
        "common_config_file_sha256": sha256(common_file),
        "allowed_config_differences": [
            "speculative_config.num_speculative_tokens",
            "worker_extension_cls",
        ],
        "native_worker": pair["native"]["worker_extension_cls"],
        "evaluation_token": secrets.token_urlsafe(48),
        "runtime_worker_observation_required": True,
    }
    write_new(root / "setup.json", setup)
    print(
        json.dumps(
            {
                "setup": str(root / "setup.json"),
                "common_config_sha256": setup["common_config_sha256"],
            }
        )
    )


def _load_setup(path: Path) -> dict[str, Any]:
    setup = read_json(path)
    if (
        setup.get("schema_version") != 1
        or path.resolve() != Path(setup["private_root"]) / "setup.json"
    ):
        raise ValueError("Invalid or relocated setup file")
    return setup


def _client_env(source: str | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VSPEC_", "HUST_", "VLLM_")) and key != "PYTHONPATH"
    }
    if source:
        env["PYTHONPATH"] = str(existing_path(source, "client source directory"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _server_env(setup: dict[str, Any], arm: str, run_dir: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HUST_", "VSPEC_", "VLLM_")) and key != "PYTHONPATH"
    }
    env.update(setup["common_env"])
    if env.get("ASCEND_VISIBLE_DEVICES") or env.get("ASCEND_RT_VISIBLE_DEVICES", "0,1") != "0,1":
        raise ValueError("Both arms require unambiguous logical NPU visibility 0,1")
    if arm == "dynamic":
        env.update(setup["dynamic_env"])
    env.update(
        VLLM_PLUGINS="ascend,vspec_eval" if arm == "native" else "ascend,vspec,vspec_eval",
        HUST_VSPEC_PRIVATE_COMPILER_ROOT=str(run_dir),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        ASCEND_RT_VISIBLE_DEVICES="0,1",
        VSPEC_EVAL_ARM=arm,
        VSPEC_EVAL_TOKEN=setup["evaluation_token"],
        PYTHONPATH=(
            os.pathsep.join(
                (
                    setup["paths"]["native_backend_package"],
                    setup["paths"]["native_evaluation_static_dir"],
                    setup["paths"]["native_evaluation_dir"],
                )
            )
            if arm == "native"
            else os.pathsep.join(
                (
                    setup["paths"]["dynamic_backend_package"],
                    setup["paths"]["dynamic_evaluation_dir"],
                )
            )
        ),
    )
    if (
        env.get("HUST_VSPEC_CAPTURE_SAFE_STATIC") != "1"
        or env.get("HUST_VSPEC_HOT_STATIC_SHAPES") != "14,42,70,98,126"
    ):
        raise ValueError("Matched static compiler settings are required in common-env.json")
    ipc = run_dir / "ipc"
    ipc.mkdir(parents=True, exist_ok=True, mode=0o700)
    env.update(TMPDIR=str(ipc), TMP=str(ipc), TEMP=str(ipc), VLLM_RPC_BASE_PATH=str(ipc))
    if arm == "dynamic":
        env["HUST_VSPEC_ENABLED"] = "1"
    for key, suffix in CACHE_DIRS.items():
        destination = run_dir / "cache" / suffix
        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(destination)
    return env


def _current_physical_npus() -> list[int]:
    output = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=15)
    rows = re.findall(r"^\s*(\d+)\s+0\s+(\d+)\s+Ascend 910B2\s*$", output, re.MULTILINE)
    if len(rows) != 2 or {logical for _, logical in rows} != {"0", "1"}:
        raise RuntimeError("Expected exactly two visible Ascend 910B2 logical NPU devices")
    physical = [int(device) for device, _ in sorted(rows, key=lambda row: int(row[1]))]
    if len(set(physical)) != 2:
        raise RuntimeError("Logical NPU devices do not map to distinct physical devices")
    return physical


def _read_ids(dataset: str, plan: Path, selection: Path | None) -> tuple[list[str], str]:
    if dataset == "mmlu-pro":
        cases = read_json(plan / "cases.json")
        ids = [row["case_id"] for row in cases]
        if selection:
            raise ValueError("MMLU-Pro selection must be a verified derived plan, not --selection")
        selection_hash = sha256(plan / "cases.json")
    else:
        if selection is None:
            raise ValueError("Judged datasets require a frozen --selection manifest")
        manifest = read_json(selection)
        if manifest.get("dataset") != dataset or manifest.get(
            "source_plan_contract_sha256"
        ) != sha256(plan / "contract.json"):
            raise ValueError("Selection does not belong to this client plan")
        ids = manifest["case_ids"]
        if len(ids) != manifest["count"] or canonical_sha256(ids) != manifest["case_ids_sha256"]:
            raise ValueError("Selection case IDs failed integrity check")
        task_ids = {row["case_id"] for row in read_json(plan / "tasks.json")}
        if not set(ids) <= task_ids:
            raise ValueError("Selection contains case IDs absent from the frozen plan")
        selection_hash = manifest["case_ids_sha256"]
    if not ids or len(ids) != len(set(ids)) or not all(isinstance(x, str) and x for x in ids):
        raise ValueError("Invalid or empty case-ID set")
    return ids, selection_hash


def _client_command(
    setup: dict[str, Any], dataset: str, plan: Path, output: Path, case_ids: list[str]
) -> list[str]:
    command = [setup["paths"]["client_python"], "-B", "-m", MODULES[dataset]]
    if dataset == "mmlu-pro":
        return command + [
            "mmlu-pro",
            "run",
            "--plan",
            str(plan),
            "--output",
            str(output),
            "--base-url",
            f"http://127.0.0.1:{setup['port']}",
            "--served-model",
            setup["served_model"],
            "--concurrency",
            "16",
        ]
    return (
        command
        + ["run", "--plan", str(plan), "--output", str(output)]
        + [item for case_id in case_ids for item in ("--case-id", case_id)]
    )


def _verify_frozen_endpoint(setup: dict[str, Any], dataset: str, plan: Path) -> None:
    if dataset == "mmlu-pro":
        return
    contract = read_json(plan / "contract.json")
    solver = contract.get("solver", {})
    endpoint = solver.get("base_url", "").rstrip("/").removesuffix("/v1")
    expected = f"http://127.0.0.1:{setup['port']}"
    if endpoint != expected or solver.get("model") != setup["served_model"]:
        raise ValueError("Frozen plan solver URL/model must match both generated server arms")


def _wait_ready(process: subprocess.Popen[Any], port: int, alias: str, timeout: int) -> float:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness with code {process.returncode}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
                models = json.load(response)
            if any(row.get("id") == alias for row in models["data"]):
                return time.monotonic() - started
        except (OSError, ValueError, KeyError, TypeError):
            pass
        time.sleep(2)
    raise TimeoutError("Server readiness deadline exceeded; generation was not timed")


def _eval_request(setup: dict[str, Any], route: str, *, post: bool = False) -> Any:
    request = urllib.request.Request(
        f"http://127.0.0.1:{setup['port']}/{route}",
        data=b"" if post else None,
        method="POST" if post else "GET",
        headers={"Authorization": "Bearer " + setup["evaluation_token"]},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=180) as response:
        return json.load(response)


def _validate_runtime_metadata(arm: str, response: Any) -> dict[str, Any]:
    rows = response.get("results") if isinstance(response, dict) else None
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("Evaluation metadata must cover both worker ranks")
    ranks = set()
    common_keys = (
        "dtype",
        "quantization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "explicit_kv_bytes",
        "prefix_cache",
        "mamba_cache_mode",
        "model_text_type",
    )
    for row in rows:
        if row["arm"] != arm or row["configured_speculative_capacity"] != (
            4 if arm == "native" else 8
        ):
            raise RuntimeError("Evaluation worker arm or capacity mismatch")
        if row["vspec_module_loaded"] != (arm == "dynamic"):
            raise RuntimeError("Evaluation worker plugin-loading state mismatch")
        if row["quantization"] is not None or "bfloat16" not in row["dtype"]:
            raise RuntimeError("Evaluation worker is not pure BF16")
        if row["explicit_kv_bytes"] <= 0 or not row["prefix_cache"]:
            raise RuntimeError("Evaluation worker KV/prefix-cache configuration differs")
        if not row["compiler"]["actual_enable_static_kernel"]:
            raise RuntimeError("Static compiler was not enabled in a worker")
        state = row["static"]
        install = state["static_install"] if arm == "native" else state["static"]
        if not install["process_isolation"] or not install["actual_installed_device_binary_sha256"]:
            raise RuntimeError("Installed static operator proof is incomplete")
        ranks.add(install["rank"])
    if ranks != {0, 1}:
        raise RuntimeError("Evaluation metadata did not cover both worker ranks")
    common = {key: rows[0][key] for key in common_keys}
    if any({key: row[key] for key in common_keys} != common for row in rows[1:]):
        raise RuntimeError("Worker ranks disagree on shared runtime configuration")
    return common


def _validate_operator_proof(arm: str, response: Any) -> None:
    rows = response.get("results") if isinstance(response, dict) else None
    if not isinstance(rows, list) or len(rows) != 2 or any(row.get("arm") != arm for row in rows):
        raise RuntimeError("Operator proof did not cover the selected two-arm worker ranks")
    if arm == "native":
        if not all(
            row.get("static_runtime", {}).get("vspec_plugin_absent")
            and row.get("static_runtime", {}).get("static_install")
            for row in rows
        ):
            raise RuntimeError("Native operator proof is incomplete")
    elif not all(
        len(row.get("native_edge_convolution", ())) == 30
        and len(row.get("native_gamma8_RGDR", ())) == 30
        for row in rows
    ):
        raise RuntimeError("Dynamic operator proof lacks the expected Qwen3.5 layers")


def _stop_owned_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _record_usage(dataset: str, output: Path, expected_ids: list[str]) -> dict[str, Any]:
    files = sorted((output / "requests").glob("*.json"))
    records = [read_json(path) for path in files]
    if len(records) != len(expected_ids) or {row.get("case_id") for row in records} != set(
        expected_ids
    ):
        raise RuntimeError("Client request records do not match the frozen selection")
    failures = 0
    tokens = 0
    statuses: Counter[str] = Counter()
    for row in records:
        status = (
            ("ok" if row.get("success") else "error")
            if dataset == "mmlu-pro"
            else row.get("status")
        )
        statuses[str(status)] += 1
        if status != "ok":
            failures += 1
            continue
        usage = row.get("usage")
        value = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if type(value) is not int or value < 0:
            raise RuntimeError("Successful request has no valid completion-token usage")
        tokens += value
    return {
        "record_count": len(records),
        "completion_tokens": tokens,
        "transport_or_response_failures": failures,
        "statuses": dict(statuses),
        "record_hashes": {path.name: sha256(path) for path in files},
    }


def run(args: argparse.Namespace) -> None:
    setup = _load_setup(existing_path(args.setup, "setup file"))
    arm, dataset = args.arm, args.dataset
    plan = existing_path(args.plan, "frozen client plan")
    selection = existing_path(args.selection, "case selection") if args.selection else None
    case_ids, selection_hash = _read_ids(dataset, plan, selection)
    _verify_frozen_endpoint(setup, dataset, plan)
    if not args.tag.isascii() or not args.tag.replace("-", "").isalnum() or len(args.tag) > 40:
        raise ValueError("Tag must contain only ASCII letters, numbers or hyphens")
    root = Path(setup["private_root"])
    run_dir = root / "runs" / dataset / arm / args.tag
    if run_dir.exists():
        raise FileExistsError("Use a new run tag; results are never overwritten")
    config = root / "server-config" / f"{arm}.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    python = setup["paths"][f"{arm}_python"]
    backend = setup["paths"][f"{arm}_backend_package"]
    vllm_cli = Path(python).parent / "vllm"
    if not vllm_cli.is_file():
        raise FileNotFoundError(f"Installed vLLM console script missing: {vllm_cli}")
    server_cmd = [
        python,
        "-m",
        "vllm_hust_vspec.run_with_backend",
        "--backend-package",
        backend,
        "--qwen35-mtp-compat",
        "--",
        str(vllm_cli),
        "serve",
        "--config",
        str(config),
    ]
    solver_out = run_dir / "solver"
    client_cmd = _client_command(setup, dataset, plan, solver_out, case_ids)
    if not args.execute:
        print(
            json.dumps(
                {"server": server_cmd, "client": client_cmd, "case_count": len(case_ids)}, indent=2
            )
        )
        return
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", setup["port"]))
    physical_npus = _current_physical_npus()
    run_dir.mkdir(parents=True, mode=0o700)
    env = _server_env(setup, arm, run_dir)
    write_new(
        run_dir / "launch.json",
        {
            "server_command": server_cmd,
            "client_command": client_cmd,
            "server_config_sha256": sha256(config),
            "common_config_sha256": setup["common_config_sha256"],
            "plan_contract_sha256": sha256(plan / "contract.json"),
            "selection_sha256": selection_hash,
            "plugin_enabled_for_server": arm == "dynamic",
            "qwen35_mtp_compat_enabled_for_both_arms": True,
            "no_saved_kv_or_answers_preloaded": True,
            "evaluation_metadata_and_cache_reset_required_before_timing": True,
            "physical_npus": physical_npus,
        },
    )
    process = None
    with (run_dir / "server.log").open("x", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(
                server_cmd,
                cwd=run_dir,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            startup_seconds = _wait_ready(
                process, setup["port"], setup["served_model"], args.startup_timeout
            )
            before = _eval_request(setup, "vspec-evaluation-metadata")
            runtime_common = _validate_runtime_metadata(arm, before)
            write_new(run_dir / "worker-before.json", before)
            reset = _eval_request(setup, "vspec-evaluation-reset-caches", post=True)
            if not all(
                reset.get(key) is True
                for key in ("prefix_cache_cleared", "encoder_cache_cleared", "mm_cache_cleared")
            ):
                raise RuntimeError("Fresh-server cache reset was not fully acknowledged")
            write_new(run_dir / "cache-reset.json", reset)
            started = time.monotonic()
            with (run_dir / "client.log").open("x", encoding="utf-8") as client_log:
                result = subprocess.run(
                    client_cmd,
                    env=_client_env(args.client_source),
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            wall = time.monotonic() - started
            usage = _record_usage(dataset, solver_out, case_ids)
            after = _eval_request(setup, "vspec-evaluation-metadata")
            if _validate_runtime_metadata(arm, after) != runtime_common:
                raise RuntimeError("Shared runtime configuration changed during the run")
            write_new(run_dir / "worker-after.json", after)
            operator = _eval_request(setup, "vspec-evaluation-operator-proof")
            _validate_operator_proof(arm, operator)
            write_new(run_dir / "operator-proof.json", operator)
            receipt = {
                "schema_version": 1,
                "dataset": dataset,
                "arm": arm,
                "case_count": len(case_ids),
                "plan_contract_sha256": sha256(plan / "contract.json"),
                "selection_sha256": selection_hash,
                "common_config_sha256": setup["common_config_sha256"],
                "server_config_sha256": sha256(config),
                "startup_seconds_excluded": startup_seconds,
                "client_wall_seconds": wall,
                "output_tokens_per_second": usage["completion_tokens"] / wall,
                "client_exit_code": result.returncode,
                "native_worker": setup["native_worker"],
                "physical_npus": physical_npus,
                "worker_runtime_common": runtime_common,
                "worker_before_sha256": sha256(run_dir / "worker-before.json"),
                "worker_after_sha256": sha256(run_dir / "worker-after.json"),
                "cache_reset_sha256": sha256(run_dir / "cache-reset.json"),
                "operator_proof_sha256": sha256(run_dir / "operator-proof.json"),
                "solver_output": str(solver_out),
                "timing_includes_client_overhead": True,
                "natural_output_lengths_may_differ_not_fixed_work": True,
                **usage,
            }
            write_new(run_dir / "solver-receipt.json", receipt)
            if result.returncode or usage["transport_or_response_failures"]:
                raise RuntimeError("Client failed; inspect private logs and request records")
            print(
                json.dumps(
                    {
                        key: receipt[key]
                        for key in (
                            "dataset",
                            "arm",
                            "case_count",
                            "completion_tokens",
                            "client_wall_seconds",
                            "output_tokens_per_second",
                        )
                    }
                )
            )
        finally:
            if process is not None:
                _stop_owned_server(process)


def score(args: argparse.Namespace) -> None:
    setup = _load_setup(existing_path(args.setup, "setup file"))
    run_dir = existing_path(args.run_dir, "completed solver run")
    receipt = read_json(run_dir / "solver-receipt.json")
    dataset = receipt["dataset"]
    plan = existing_path(args.plan, "frozen client plan")
    if receipt["plan_contract_sha256"] != sha256(plan / "contract.json"):
        raise ValueError("Score plan differs from solver plan")
    if (run_dir / "score-receipt.json").exists():
        raise FileExistsError("This solver run has already been scored")
    prefix = [setup["paths"]["client_python"], "-B", "-m", MODULES[dataset]]
    summary_path = run_dir / "summary"
    if dataset == "mmlu-pro":
        command = prefix + [
            "mmlu-pro",
            "summary",
            "--plan",
            str(plan),
            "--accuracy-dir",
            receipt["solver_output"],
            "--out",
            str(run_dir / "mmlu-summary.json"),
        ]
        with (run_dir / "summary.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                command,
                env=_client_env(args.client_source),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = read_json(run_dir / "mmlu-summary.json")
        if not summary["complete"]:
            raise RuntimeError("Partial MMLU-Pro summary is not a score")
        quality = {"overall": summary["accuracy"]["accuracy"], "unresolved": 0}
    else:
        grades = run_dir / "judge"
        judge_cmd = prefix + [
            "judge",
            "--plan",
            str(plan),
            "--responses",
            receipt["solver_output"],
            "--output",
            str(grades),
        ]
        with (run_dir / "judge.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                judge_cmd,
                env=_client_env(args.client_source),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary_cmd = prefix + [
            "summary",
            "--plan",
            str(plan),
            "--responses",
            receipt["solver_output"],
            "--grades",
            str(grades),
            "--output",
            str(summary_path),
        ]
        with (run_dir / "summary.log").open("x", encoding="utf-8") as log:
            subprocess.run(
                summary_cmd,
                env=_client_env(args.client_source),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        summary = read_json(summary_path / "summary.json")
        track = "gold" if dataset == "hle-verified" else "olympiad"
        stats = summary["tracks"][track]
        if stats["total"] != receipt["case_count"] and dataset == "hle-verified":
            raise RuntimeError("HLE Gold score does not cover the complete solver selection")
        quality = {
            "track": track,
            "successes": stats["successes"],
            "total": stats["total"],
            "unresolved": stats.get("unresolved", stats.get("judge_unresolved", 0)),
            "observed_lower_bound": stats["successes"] / stats["total"],
            "scoring_policy": summary["scoring_policy"],
        }
        if quality["total"] <= 0 or quality["successes"] + quality["unresolved"] > quality["total"]:
            raise RuntimeError("Judge summary contains invalid success/unresolved counts")
    result = {
        "schema_version": 1,
        "dataset": dataset,
        "arm": receipt["arm"],
        "plan_contract_sha256": receipt["plan_contract_sha256"],
        "selection_sha256": receipt["selection_sha256"],
        "quality": quality,
        "summary_sha256": sha256(
            run_dir / "mmlu-summary.json"
            if dataset == "mmlu-pro"
            else summary_path / "summary.json"
        ),
    }
    write_new(run_dir / "score-receipt.json", result)
    print(json.dumps(result, ensure_ascii=False))


def compare(args: argparse.Namespace) -> None:
    runs = {}
    for arm, path in (("native", args.native_run), ("dynamic", args.dynamic_run)):
        directory = existing_path(path, f"{arm} run")
        solver = read_json(directory / "solver-receipt.json")
        score_result = read_json(directory / "score-receipt.json")
        if solver["arm"] != arm or score_result["arm"] != arm:
            raise ValueError("Run arm label mismatch")
        if solver["client_exit_code"] or solver["transport_or_response_failures"]:
            raise ValueError("Cannot compare a failed solver run")
        if (
            score_result["dataset"] != solver["dataset"]
            or score_result["selection_sha256"] != solver["selection_sha256"]
        ):
            raise ValueError("Score receipt differs from solver scope")
        for name, key in (
            ("worker-before.json", "worker_before_sha256"),
            ("worker-after.json", "worker_after_sha256"),
            ("cache-reset.json", "cache_reset_sha256"),
            ("operator-proof.json", "operator_proof_sha256"),
        ):
            if solver.get(key) != sha256(directory / name):
                raise ValueError(f"Missing or altered {arm} runtime proof: {name}")
        runs[arm] = (solver, score_result)
    native, native_score = runs["native"]
    dynamic, dynamic_score = runs["dynamic"]
    for key in (
        "dataset",
        "case_count",
        "plan_contract_sha256",
        "selection_sha256",
        "common_config_sha256",
        "physical_npus",
        "worker_runtime_common",
    ):
        if native[key] != dynamic[key]:
            raise ValueError(f"Paired runs disagree on {key}")
    if native_score["plan_contract_sha256"] != dynamic_score["plan_contract_sha256"]:
        raise ValueError("Scoring plans differ")
    if not native.get("native_worker"):
        raise ValueError("No installed native observer worker was configured; diagnostic only")
    dataset = native["dataset"]
    if dataset == "mmlu-pro":
        quality_native = native_score["quality"]["overall"]
        quality_dynamic = dynamic_score["quality"]["overall"]
        worst_delta = quality_dynamic - quality_native
        unresolved = False
    else:
        n, d = native_score["quality"], dynamic_score["quality"]
        if (
            n["track"] != d["track"]
            or n["total"] != d["total"]
            or n["scoring_policy"] != d["scoring_policy"]
        ):
            raise ValueError("Scoring scope/policy differs")
        quality_native = n["observed_lower_bound"]
        quality_dynamic = d["observed_lower_bound"]
        worst_delta = quality_dynamic - (n["successes"] + n["unresolved"]) / n["total"]
        unresolved = bool(n["unresolved"] or d["unresolved"])
    if native["output_tokens_per_second"] <= 0:
        raise ValueError("Native throughput must be positive")
    ratio = dynamic["output_tokens_per_second"] / native["output_tokens_per_second"]
    if not math.isfinite(ratio):
        raise ValueError("Non-finite throughput ratio")
    result = {
        "dataset": dataset,
        "case_count": native["case_count"],
        "output_throughput_ratio_dynamic_over_native": ratio,
        "quality_native_observed": quality_native,
        "quality_dynamic_observed": quality_dynamic,
        "quality_worst_case_delta": worst_delta,
        "unresolved_grades_present": unresolved,
        "throughput_ge_1_10": ratio >= 1.10,
        "quality_worst_case_within_2pp": worst_delta >= -0.02,
        "independent_runtime_worker_and_operator_proof_still_required": True,
        "natural_output_lengths_may_differ_not_fixed_work": True,
    }
    if args.output:
        output = private_root(args.output)
        write_new(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="freeze matched private server configs")
    for name in (
        "private-root",
        "model-dir",
        "common-config",
        "common-env",
        "dynamic-env",
        "native-python",
        "dynamic-python",
        "client-python",
        "native-backend-package",
        "dynamic-backend-package",
        "native-evaluation-dir",
        "dynamic-evaluation-dir",
        "served-model",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--port", type=int, required=True)
    p.set_defaults(fn=prepare)
    p = sub.add_parser("run", help="start one arm, time one frozen solver selection, then stop it")
    p.add_argument("--setup", required=True)
    p.add_argument("--arm", choices=("native", "dynamic"), required=True)
    p.add_argument("--dataset", choices=tuple(MODULES), required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--selection", help="required for HLE-Verified and FrontierScience")
    p.add_argument("--tag", required=True)
    p.add_argument(
        "--client-source", help="fresh installed client source; do not use an old experiment SOURCE"
    )
    p.add_argument("--startup-timeout", type=int, default=3600)
    p.add_argument("--execute", action="store_true", help="without this flag print commands only")
    p.set_defaults(fn=run)
    p = sub.add_parser("score", help="judge separately after solver server stops, then summarize")
    p.add_argument("--setup", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--client-source")
    p.set_defaults(fn=score)
    p = sub.add_parser("compare", help="check paired scope and print throughput/quality")
    p.add_argument("--native-run", required=True)
    p.add_argument("--dynamic-run", required=True)
    p.add_argument("--output", help="new private JSON file")
    p.set_defaults(fn=compare)
    return parser


def main(argv: list[str] | None = None) -> None:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
