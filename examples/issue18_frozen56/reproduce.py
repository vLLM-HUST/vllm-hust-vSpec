"""Reproduce the frozen issue #18 MMLU-Pro 56-question offline workload.

The test/validation parquet files, frozen ID manifest, model and engine config
are supplied by the operator.  No dataset or model content is bundled here.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any


CHOICES = "ABCDEFGHIJ"
MAX_OUTPUT_TOKENS = 4000
SHORT_WARMUP_TOKENS = 16
TEST_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
VALIDATION_SHA256 = "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078"
MANIFEST_SHA256 = "d4952f9c47c31ea3c72234c642f8247ba581553061aa650766746435c3a932b2"


def configure_environment(args: argparse.Namespace) -> None:
    profile = json.loads(args.env_profile.read_text(encoding="utf-8"))
    entries = {**profile["common"], **profile[args.arm]}
    allowed = ("HUST_", "VLLM_", "VSPEC_", "HCCL_", "OMP_", "PYTORCH_NPU_", "BENCH_")
    for key, value in entries.items():
        if not key.startswith(allowed) or not isinstance(value, str) or "/" in value:
            raise ValueError(f"Unsafe environment profile entry: {key}")
        os.environ[key] = value
    site = args.installed_site_packages.resolve(strict=True)
    package = site / "vllm_hust_vspec"
    backend = site / "vllm_ascend"
    if not package.is_dir() or not backend.is_dir():
        raise FileNotFoundError("Installed plugin/backend package is incomplete")
    compat = package / "_vendor/qwen35_mtp_compat"
    evaluation = package / "evaluation"
    native_static = package / "evaluation_static"
    cann_home_text = os.environ.get("ASCEND_HOME_PATH")
    if not cann_home_text:
        raise ValueError("Source the matching CANN environment before running")
    cann_home = Path(cann_home_text).resolve(strict=True)
    search = [compat, evaluation]
    if args.arm == "native":
        search.append(native_static)
    search.extend((site, cann_home / "python/site-packages",
                   cann_home / "opp/built-in/op_impl/ai_core/tbe"))
    if not all(path.is_dir() for path in search):
        raise FileNotFoundError("Qwen3.5 compatibility or CANN Python path is missing")
    os.environ["PYTHONPATH"] = os.pathsep.join(map(str, search))
    # multiprocessing spawn copies the parent's sys.path, not just PYTHONPATH.
    for path in reversed(search):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    custom_opp = backend / "_cann_ops_custom/vendors/custom_transformer"
    if not custom_opp.is_dir():
        raise FileNotFoundError("Installed custom Ascend operators are missing")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp)
    os.environ["QWEN35_MTP_COMPAT"] = "1"
    os.environ["HUST_VSPEC_RUN_PACKAGE_WORKER"] = "1"
    if args.arm == "vspec":
        if args.runtime_env_file is None:
            raise ValueError("vSpec requires the installed operator runtime.env")
        for line in args.runtime_env_file.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = shlex.split(line)
            if len(parts) != 2 or parts[0] != "export" or "=" not in parts[1]:
                raise ValueError("Unexpected runtime.env assignment")
            key, value = parts[1].split("=", 1)
            if not key.startswith("VSPEC_"):
                raise ValueError(f"Unexpected runtime.env key: {key}")
            os.environ[key] = value
    # Private caches are new for every output directory.  Keep IPC paths short.
    cache = args.output / "cache"
    for key, relative in {
        "XDG_CACHE_HOME": "xdg", "XDG_CONFIG_HOME": "xdg-config",
        "XDG_DATA_HOME": "xdg-data", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TRITON_CACHE_DIR": "triton", "TORCH_EXTENSIONS_DIR": "torch-extensions",
        "VLLM_CACHE_ROOT": "vllm", "VLLM_CONFIG_ROOT": "vllm-config",
        "VLLM_ASSETS_CACHE": "vllm-assets", "VLLM_XLA_CACHE_PATH": "vllm-xla",
        "ACL_OP_COMPILER_CACHE_DIR": "acl", "ASCEND_WORK_PATH": "ascend-work",
        "ASCEND_PROCESS_LOG_PATH": "ascend-logs", "NUMBA_CACHE_DIR": "numba",
        "PYTHONPYCACHEPREFIX": "pycache",
    }.items():
        target = cache / relative
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["HUST_VSPEC_PRIVATE_COMPILER_ROOT"] = str(args.output)
    os.environ["TMPDIR"] = tempfile.mkdtemp(prefix="vspec-issue18-", dir="/tmp")
    os.environ["PYTHONHASHSEED"] = "0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def validate_static_compile_path(output: Path) -> None:
    # This CANN stack encodes its absolute build path in a single generated
    # filename. Leave margin for operator-specific suffixes before allocating
    # NPU resources; the observed failing run path was 117 bytes.
    if len(os.fsencode(str(output.resolve()))) > 96:
        raise ValueError("Output path is too long for CANN static compilation; use a short --output")


def reuse_compiler_cache(source: Path, output: Path) -> None:
    """Copy a prior run's compiler cache without modifying the source run."""
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Cache source and output must be separate directories")
    for relative in ("cache/triton", "static_kernel_compile_outputs"):
        if not (source / relative).is_dir():
            raise FileNotFoundError(source / relative)
    for relative in ("cache", "static_kernel_compile_outputs"):
        original = source / relative
        if original.is_dir():
            shutil.copytree(original, output / relative)
    repair_triton_group_paths(output / "cache/triton")


def repair_triton_group_paths(cache: Path) -> None:
    for manifest in cache.glob("*/__grp__*.json"):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        paths = payload.get("child_paths")
        if not isinstance(paths, dict):
            continue
        for name in paths:
            artifact = manifest.parent / name
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            paths[name] = str(artifact)
        manifest.write_text(json.dumps(payload), encoding="utf-8")


def seed_triton_kernels(source: Path, output: Path) -> list[str]:
    """Diagnostic only: isolate formal-time Triton compilation from other caches.

    Copy the five variants first observed during a short-pilot formal run.
    The destination still starts with fresh graph/static/CANN caches. This is
    deliberately not a fresh-install reproduction or a distributable fix.
    """
    wanted = {
        "postprocess_mamba_fused_kernel",
        "batch_memcpy_kernel",
        "gdn_gid123_full_stable_refresh",
    }
    source_cache = source / "cache/triton"
    destination = output / "cache/triton"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for artifact in source_cache.glob("*/*.npubin"):
        if artifact.stem not in wanted:
            continue
        shutil.copytree(artifact.parent, destination / artifact.parent.name)
        copied.append(artifact.stem)
    if not copied or set(copied) != wanted:
        raise ValueError("Source does not contain all diagnostic Triton kernels")
    repair_triton_group_paths(destination)
    return copied


def clean(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["options"] = [option for option in row["options"] if option != "N/A"]
    return row


def format_example(question: str, options: list[str], cot: str = "") -> str:
    if not cot:
        cot = "Let's think step by step."
    if cot.startswith("A: "):
        cot = cot[3:]
    text = f"Question: {question}\nOptions: "
    text += "".join(f"{CHOICES[i]}. {option}\n" for i, option in enumerate(options))
    return text + f"Answer: {cot}\n\n"


def make_prompt(row: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    text = (
        "The following are multiple choice questions (with answers) about {}. "
        "Think step by step and then output the answer in the format of "
        '"The answer is (X)" at the end.\n\n'
    ).format(row["category"])
    for shot in shots:
        text += format_example(shot["question"], shot["options"], shot["cot_content"])
    return text + format_example(row["question"], row["options"])


def extract_answer(text: str) -> str | None:
    text = text.replace("**", "")
    match = re.search(r"answer is \(?([A-J])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[aA]nswer:\s*([A-J])", text)
    if match:
        return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    for path, expected in ((args.test_parquet, TEST_SHA256),
                           (args.validation_parquet, VALIDATION_SHA256)):
        if sha256(path) != expected:
            raise ValueError(f"Pinned MMLU-Pro parquet checksum mismatch: {path}")
    if sha256(args.manifest) != MANIFEST_SHA256:
        raise ValueError("Frozen issue #18 manifest checksum mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ids = manifest["performance_ids"]
    pilot_ids = set(manifest["excluded_pilot_ids"])
    if len(ids) != 56 or len(set(ids)) != 56 or len(pilot_ids) != 14 or set(ids) & pilot_ids:
        raise ValueError("Expected disjoint frozen 56 scored and 14 pilot IDs")
    test = [clean(row) for row in pq.read_table(args.test_parquet).to_pylist()]
    validation = [clean(row) for row in pq.read_table(args.validation_parquet).to_pylist()]
    if len(test) != 12032 or len(validation) != 70:
        raise ValueError("Pinned MMLU-Pro split cardinality changed")
    by_id = {row["question_id"]: row for row in test}
    if len(by_id) != len(test) or any(i not in by_id for i in [*ids, *pilot_ids]):
        raise ValueError("Frozen ID missing or duplicated")
    shots_by_category: dict[str, list[dict[str, Any]]] = {}
    for row in validation:
        shots_by_category.setdefault(row["category"], []).append(row)
    if len(shots_by_category) != 14 or any(len(shots) != 5 for shots in shots_by_category.values()):
        raise ValueError("Expected five validation shots in each of 14 subjects")
    if len({by_id[i]["category"] for i in pilot_ids}) != 14:
        raise ValueError("Pilot IDs must cover 14 subjects")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)

    def one(question_id: int) -> dict[str, Any]:
        row = by_id[question_id]
        if row["answer"] != CHOICES[row["answer_index"]]:
            raise ValueError(f"Answer/index mismatch: {question_id}")
        prompt = make_prompt(row, shots_by_category[row["category"]])
        prompt_ids = (
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True, return_dict=False,
            ) if args.chat_template else tokenizer(prompt)["input_ids"]
        )
        if len(prompt_ids) + MAX_OUTPUT_TOKENS > 16384:
            raise ValueError(f"Prompt exceeds frozen context: {question_id}")
        return {
            "question_id": question_id,
            "category": row["category"],
            "answer": row["answer"],
            "prompt_ids": prompt_ids,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_tokens": len(prompt_ids),
        }

    cases = [one(question_id) for question_id in ids]
    if {category: sum(case["category"] == category for case in cases)
        for category in shots_by_category} != {category: 4 for category in shots_by_category}:
        raise ValueError("Expected four scored questions in each subject")
    pilot = [one(question_id) for question_id in sorted(pilot_ids, key=lambda i: by_id[i]["category"])]
    return cases, pilot


async def run_batch(engine: Any, cases: list[dict[str, Any]], max_tokens: int,
                    output: Path) -> tuple[list[dict[str, Any]], float]:
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with output.open("x", encoding="utf-8") as stream:
        async def one(case: dict[str, Any]) -> None:
            request_started = time.monotonic()
            final = None
            request_id = f"{'warmup' if max_tokens == SHORT_WARMUP_TOKENS else 'perf'}-{case['question_id']}"
            async for item in engine.generate(
                TokensPrompt(prompt_token_ids=case["prompt_ids"]), params, request_id=request_id,
            ):
                final = item
            if final is None or not final.finished or len(final.outputs) != 1:
                raise RuntimeError(f"Incomplete output: {request_id}")
            completion = final.outputs[0]
            prediction = extract_answer(completion.text)
            row = {
                "question_id": case["question_id"],
                "category": case["category"],
                "prompt_sha256": case["prompt_sha256"],
                "prompt_tokens": case["prompt_tokens"],
                "answer": case["answer"],
                "prediction": prediction,
                "correct": prediction == case["answer"],
                "unparsed": prediction is None,
                "output_text": completion.text,
                "output_token_ids": list(completion.token_ids),
                "output_tokens": len(completion.token_ids),
                "truncated": completion.finish_reason == "length",
                "finish_reason": completion.finish_reason,
                "stop_reason": completion.stop_reason,
                "latency_seconds": time.monotonic() - request_started,
                "preemptions": getattr(final.metrics, "num_preemptions", 0) or 0,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()  # Completion-order write is inside the timed region.

        await asyncio.gather(*(one(case) for case in cases))
    elapsed = time.monotonic() - started
    if len(rows) != len(cases) or {row["question_id"] for row in rows} != {
        case["question_id"] for case in cases
    }:
        raise RuntimeError("Incomplete or duplicated batch")
    return rows, elapsed


async def execute(args: argparse.Namespace, cases: list[dict[str, Any]],
                  pilot: list[dict[str, Any]]) -> None:
    if args.arm == "native" and os.environ.get("BENCH_NATIVE_KV_COMPAT") == "1":
        from vllm_hust_vspec.backends.kv_cache import apply_multi_layer_kv_cache_patch
        from vllm_hust_vspec.backends.mamba_compat import apply_mamba_runtime_compatibility_patch

        apply_multi_layer_kv_cache_patch()
        apply_mamba_runtime_compatibility_patch()
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    source = json.loads(args.engine_config.read_text(encoding="utf-8"))
    key = "q35-mtp4" if args.arm == "native" else "q35-adaptive"
    if "common" in source:
        config = {**source["common"], **source[args.arm]}
    else:
        config = dict(source[key] if key in source else source)
    config["model"] = str(args.model_dir)
    config["mamba_cache_mode"] = "align"
    if args.worker_extension:
        config["worker_extension_cls"] = args.worker_extension
    if config.get("max_num_seqs") != 16 or config.get("max_model_len") != 16384:
        raise ValueError("Engine must use max_num_seqs=16 and max_model_len=16384")
    write_json(args.output / "resolved-engine-config.json", config)
    startup_start = time.monotonic()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**config))
    write_json(args.output / "startup-summary.json", {
        "service_ready_seconds": time.monotonic() - startup_start,
        "excludes_prompt_preparation": True,
    })
    try:
        short, short_seconds = await run_batch(
            engine, pilot, SHORT_WARMUP_TOKENS, args.output / "warmup-results.jsonl"
        )
        write_json(args.output / "warmup-summary.json", {
            "n": len(short), "seconds": short_seconds,
            "output_tokens": sum(row["output_tokens"] for row in short),
        })
        if args.prime_pilot16x1024:
            # Additional, unscored priming is NOT part of the frozen issue protocol.
            extra = [dict(pilot[i], question_id=-(i + 1)) for i in range(2)]
            primed, prime_seconds = await run_batch(
                engine, [*pilot, *extra], 1024, args.output / "prime-results.jsonl"
            )
            write_json(args.output / "prime-summary.json", {
                "n": len(primed), "seconds": prime_seconds,
                "output_tokens": sum(row["output_tokens"] for row in primed),
                "protocol_extension": True,
            })
        rows, seconds = await run_batch(
            engine, cases, MAX_OUTPUT_TOKENS, args.output / "results.jsonl"
        )
        total = sum(row["output_tokens"] for row in rows)
        summary = {
            "n": len(rows), "correct": sum(row["correct"] for row in rows),
            "unparsed": sum(row["unparsed"] for row in rows),
            "truncated": sum(row["truncated"] for row in rows),
            "preemptions": sum(row["preemptions"] for row in rows),
            "output_tokens": total, "generation_wall_seconds": seconds,
            "output_tokens_per_second": total / seconds,
            "prime_pilot16x1024": args.prime_pilot16x1024,
            "by_subject": {
                category: {
                    "n": sum(row["category"] == category for row in rows),
                    "correct": sum(row["correct"] for row in rows if row["category"] == category),
                }
                for category in sorted({row["category"] for row in rows})
            },
            "median_request_latency_seconds": statistics.median(row["latency_seconds"] for row in rows),
        }
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--engine-config", type=Path, required=True)
    parser.add_argument("--env-profile", type=Path, required=True)
    parser.add_argument("--installed-site-packages", type=Path, required=True)
    parser.add_argument("--runtime-env-file", type=Path)
    parser.add_argument("--arm", choices=("native", "vspec"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-extension")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--prime-pilot16x1024", action="store_true")
    parser.add_argument("--reuse-cache-from", type=Path,
                        help="Copy compiler artifacts from a prior run; marks this run as warm-cache")
    parser.add_argument("--seed-triton-kernels-from", type=Path,
                        help="Diagnostic: seed only three families of Triton kernels, not graph/CANN caches")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for name in ("test_parquet", "validation_parquet", "manifest", "model_dir",
                 "engine_config", "env_profile", "installed_site_packages"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.runtime_env_file is not None:
        args.runtime_env_file = args.runtime_env_file.resolve(strict=True)
    if args.reuse_cache_from is not None:
        args.reuse_cache_from = args.reuse_cache_from.resolve(strict=True)
    if args.seed_triton_kernels_from is not None:
        args.seed_triton_kernels_from = args.seed_triton_kernels_from.resolve(strict=True)
    if args.reuse_cache_from is not None and args.seed_triton_kernels_from is not None:
        raise ValueError("Choose only one cache intervention")
    args.output = args.output.resolve()
    validate_static_compile_path(args.output)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.reuse_cache_from is not None:
        reuse_compiler_cache(args.reuse_cache_from, args.output)
    seeded = []
    if args.seed_triton_kernels_from is not None:
        seeded = seed_triton_kernels(args.seed_triton_kernels_from, args.output)
    configure_environment(args)
    expected = "ascend" if args.arm == "native" else "ascend,vspec"
    if os.environ.get("VLLM_PLUGINS") != expected:
        raise RuntimeError(f"Profile must set VLLM_PLUGINS={expected} for {args.arm}")
    cases, pilot = prepare(args)
    write_json(args.output / "receipt.json", {
        "arm": args.arm,
        "chat_template": args.chat_template,
        "prime_pilot16x1024": args.prime_pilot16x1024,
        "reuse_cache_from": (str(args.reuse_cache_from)
                             if args.reuse_cache_from is not None else None),
        "seed_triton_kernels_from": (str(args.seed_triton_kernels_from)
                                    if args.seed_triton_kernels_from is not None else None),
        "seeded_triton_kernel_families": sorted(set(seeded)),
        "manifest_sha256": sha256(args.manifest),
        "test_sha256": sha256(args.test_parquet),
        "validation_sha256": sha256(args.validation_parquet),
        "engine_config_sha256": sha256(args.engine_config),
        "env_profile_sha256": sha256(args.env_profile),
        "model_dir": str(args.model_dir),
        "worker_extension": args.worker_extension,
        "scope": "frozen issue #18 performance56, not full MMLU-Pro accuracy",
    })
    write_json(args.output / "cases.json", [
        {key: row[key] for key in ("question_id", "prompt_sha256", "prompt_tokens")}
        for row in cases
    ])
    write_json(args.output / "pilot.json", [
        {key: row[key] for key in ("question_id", "prompt_sha256", "prompt_tokens")}
        for row in pilot
    ])
    if args.prepare_only:
        print(json.dumps({"prepared": 56, "pilot": 14, "output": str(args.output)}))
        return
    # The installed static worker intentionally requires cwd to be its private
    # compilation root; match the validated private launcher contract.
    os.chdir(args.output)
    asyncio.run(execute(args, cases, pilot))


if __name__ == "__main__":
    main()
