"""Runtime ABI checks for the supported vLLM-HUST host trees."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

TESTED_VLLM_REVISION = "762f85b311fbab0bcf8921dd216f5093cd58b9b8"
TESTED_ASCEND_REVISION = "4e57439e58ed3d78e675f9fd7b4614fb183c5394"

COMMON_REQUIREMENTS = (
    ("vllm.v1.outputs", None, ("SamplerOutput",)),
    ("vllm_ascend.compilation.acl_graph", None, ("ACLGraphWrapper",)),
)

DRAFT_REQUIREMENTS = (
    (
        "vllm.v1.spec_decode.draft_model",
        None,
        ("DraftModelProposer",),
    ),
    (
        "vllm.v1.spec_decode.utils",
        None,
        ("compute_new_slot_mapping",),
    ),
    (
        "vllm_ascend.spec_decode.draft_proposer",
        None,
        ("AscendDraftModelProposer",),
    ),
    (
        "vllm_ascend.spec_decode.llm_base_proposer",
        "AscendSpecDecodeBaseProposer",
        (
            "dummy_run",
            "_run_merged_draft",
            "attn_update_stack_num_spec_norm",
            "_propose",
            "_update_full_graph_params",
        ),
    ),
    (
        "vllm_ascend.worker.model_runner_v1",
        "NPUModelRunner",
        ("_pad_query_start_loc_for_fia",),
    ),
)

EAGLE_REQUIREMENTS = (
    (
        "vllm_ascend.sample.rejection_sampler",
        "AscendRejectionSampler",
        ("forward",),
    ),
    (
        "vllm_ascend.spec_decode.eagle_proposer",
        "AscendEagleProposer",
        ("propose",),
    ),
    (
        "vllm_ascend.spec_decode.llm_base_proposer",
        "AscendSpecDecodeBaseProposer",
        (
            "_get_model",
            "_maybe_share_lm_head",
            "dummy_run",
            "load_model",
            "_propose",
            "_update_full_graph_params",
        ),
    ),
    (
        "vllm_ascend.worker.model_runner_v1",
        "NPUModelRunner",
        ("load_model", "_sample", "_update_full_graph_params_if_needed"),
    ),
)

QWEN2_EAGLE_REQUIREMENTS = (
    (
        "vllm",
        "ModelRegistry",
        ("get_supported_archs", "register_model"),
    ),
)

DFLASH_REQUIREMENTS = (
    (
        "vllm_ascend.spec_decode.dflash_proposer",
        "AscendDflashProposer",
        ("dummy_run", "_propose"),
    ),
)

ADAPTIVE_REQUIREMENTS = (
    (
        "vllm.v1.core.sched.scheduler",
        "Scheduler",
        (
            "schedule",
            "make_spec_decoding_stats",
            "update_from_output",
            "update_draft_token_ids",
        ),
    ),
    (
        "vllm.v1.cudagraph_dispatcher",
        "CudagraphDispatcher",
        (
            "dispatch",
            "initialize_cudagraph_keys",
            "_get_lora_cases",
            "_create_padded_batch_descriptor",
            "add_cudagraph_key",
        ),
    ),
    (
        "vllm.v1.worker.gpu_model_runner",
        "AsyncGPUModelRunnerOutput",
        ("get_output",),
    ),
    (
        "vllm_ascend.compilation.acl_graph",
        None,
        ("GraphParams", "_graph_params", "_draft_graph_params"),
    ),
    (
        "vllm_ascend.worker.model_runner_v1",
        "NPUModelRunner",
        (
            "_prepare_inputs",
            "execute_model",
            "sample_tokens",
            "_warmup_and_capture",
            "_determine_batch_execution_and_padding",
            "_copy_draft_token_ids_to_cpu",
            "propose_draft_token_ids",
            "profile_cudagraph_memory",
            "capture_model",
            "_check_and_update_cudagraph_mode",
        ),
    ),
)


@dataclass(frozen=True)
class AbiCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class SourceState:
    version: str
    revision: str | None
    dirty: bool | None
    tested_revision: str


@dataclass(frozen=True)
class HostCompatibilityReport:
    method: str
    vllm: SourceState
    vllm_ascend: SourceState
    checks: tuple[AbiCheck, ...]

    @property
    def compatible(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        report = asdict(self)
        report["compatible"] = self.compatible
        return report


def _git_source_state(module: Any, tested_revision: str) -> SourceState:
    version = str(getattr(module, "__version__", "unknown"))
    if version == "unknown":
        distribution_name = {
            "vllm": "vllm",
            "vllm_ascend": "vllm-ascend",
        }.get(getattr(module, "__name__", ""))
        if distribution_name is not None:
            try:
                version = distribution_version(distribution_name)
            except PackageNotFoundError:
                pass
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return SourceState(version, None, None, tested_revision)
    module_path = Path(module_file).resolve()
    try:
        root_result = subprocess.run(
            ["git", "-C", str(module_path.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = root_result.stdout.strip()
        revision_result = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty_result = subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return SourceState(version, None, None, tested_revision)
    return SourceState(
        version,
        revision_result.stdout.strip(),
        bool(dirty_result.stdout.strip()),
        tested_revision,
    )


def _check_requirement(
    module_name: str,
    owner_name: str | None,
    attributes: tuple[str, ...],
) -> list[AbiCheck]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return [AbiCheck(module_name, False, f"import failed: {exc}")]
    owner = module if owner_name is None else getattr(module, owner_name, None)
    owner_label = module_name if owner_name is None else f"{module_name}.{owner_name}"
    if owner is None:
        return [AbiCheck(owner_label, False, "owner is missing")]
    return [
        AbiCheck(
            f"{owner_label}.{attribute}",
            hasattr(owner, attribute),
            "available" if hasattr(owner, attribute) else "attribute is missing",
        )
        for attribute in attributes
    ]


def inspect_host_compatibility(
    method: str,
    *,
    adaptive: bool = False,
) -> HostCompatibilityReport:
    normalized_method = "draft_model" if method == "draft" else method
    if normalized_method not in {
        "draft_model",
        "eagle",
        "eagle3",
        "dflash",
    }:
        raise ValueError(f"unsupported vSpec method: {method}")

    import vllm
    import vllm_ascend

    requirements = list(COMMON_REQUIREMENTS)
    if normalized_method == "draft_model":
        requirements.extend(DRAFT_REQUIREMENTS)
    elif normalized_method in {"eagle", "eagle3"}:
        requirements.extend(EAGLE_REQUIREMENTS)
        if normalized_method == "eagle":
            requirements.extend(QWEN2_EAGLE_REQUIREMENTS)
    else:
        requirements.extend(DFLASH_REQUIREMENTS)
    if adaptive:
        requirements.extend(ADAPTIVE_REQUIREMENTS)

    checks = tuple(
        check
        for module_name, owner_name, attributes in requirements
        for check in _check_requirement(module_name, owner_name, attributes)
    )
    return HostCompatibilityReport(
        method=normalized_method,
        vllm=_git_source_state(vllm, TESTED_VLLM_REVISION),
        vllm_ascend=_git_source_state(vllm_ascend, TESTED_ASCEND_REVISION),
        checks=checks,
    )


def require_host_compatibility(
    method: str,
    *,
    adaptive: bool = False,
) -> HostCompatibilityReport:
    report = inspect_host_compatibility(method, adaptive=adaptive)
    failures = [check.name for check in report.checks if not check.ok]
    if failures:
        raise RuntimeError("vSpec host ABI is incompatible; missing: " + ", ".join(failures))
    return report


def _format_source(name: str, state: SourceState) -> str:
    revision = state.revision or "unknown"
    dirty = "unknown" if state.dirty is None else str(state.dirty).lower()
    tested = revision == state.tested_revision
    return f"{name}: version={state.version} revision={revision} dirty={dirty} tested_base={tested}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-vspec-doctor",
        description="Validate the vLLM-HUST host ABI used by vSpec.",
    )
    parser.add_argument(
        "--method",
        choices=("draft", "draft_model", "eagle", "eagle3", "dflash"),
        default="draft",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="also validate the vSpec Adaptive scheduler and graph ABI",
    )
    options = parser.parse_args(argv)
    report = inspect_host_compatibility(options.method, adaptive=options.adaptive)
    if options.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"method: {report.method}")
        print(_format_source("vllm", report.vllm))
        print(_format_source("vllm_ascend", report.vllm_ascend))
        for check in report.checks:
            status = "PASS" if check.ok else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
        print("compatible:", str(report.compatible).lower())
    if not report.compatible:
        raise SystemExit(1)
