"""Fail-closed, sampled host-updater parity for Q35 Eagle multi-step metadata.

The original Ascend updater runs on an allowlisted shadow proposer, never on
the live proposer.  This diagnostic is intentionally expensive and must only
run outside graph capture.  Neither a failed source pin nor an unsupported
metadata value may be treated as parity success.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import marshal
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import CodeType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .experimental_paths import artifact_dir

_ASCEND_ROOT = artifact_dir('VSPEC_ASCEND_SOURCE_ROOT')
_VLLM_ROOT = artifact_dir('VSPEC_VLLM_SOURCE_ROOT')
_UPSTREAM_PATH = _ASCEND_ROOT / 'vllm_ascend/spec_decode/llm_base_proposer.py'
_UPSTREAM_SHA256 = "51943285c4af8ab229e5b2db00334493cbcfd65278776f205c191ab757ccafcd"
_NATIVE_MODULE = "vllm_ascend.spec_decode.llm_base_proposer"
_NATIVE_QUALNAME = "AscendSpecDecodeBaseProposer.attn_update_stack_num_spec_norm"
_UPDATER_PARAMETERS = (
    "self", "draft_index", "old_common_metadata", "batch_size",
    "input_batch_size", "used_update_positions", "aclgraph_runtime_mode",
    "ori_seq_len", "ori_seq_len_cpu", "slot_indices", "mtp_slot_mapping",
    "attn_group",
)
_BUILDER_SOURCES = {
    _ASCEND_ROOT / 'vllm_ascend/attention/attention_v1.py':
        "48376a70ce2954bd12d08ba7a2078a0cb69faaf47e5e1c6baf6d348956981043",
    _ASCEND_ROOT / 'vllm_ascend/attention/attention_mask.py':
        "727d220386b0e8e3fe29d61ac4d4cc3d05b566845d16c7eb2843ae416a226625",
    _ASCEND_ROOT / 'vllm_ascend/attention/utils.py':
        "b600a6db861ade066920d1b5666b1c56d4cd0e5015eb63822a94d133eaae9ee5",
    _VLLM_ROOT / 'vllm/v1/attention/backend.py':
        "40b05381736376b130202bfc82d8028f81fec10e23970e579ddcf93e56a3fa5b",
}
_NATIVE_BUILDER_PATH = _ASCEND_ROOT / 'vllm_ascend/attention/attention_v1.py'
_NATIVE_BUILDER_MODULE = "vllm_ascend.attention.attention_v1"
_NATIVE_BUILDER_QUALNAME = "AscendAttentionMetadataBuilder.build"
_NATIVE_BUILDER_CODE_SHA256 = "7869c7f64ab9368e57da8ebb0443ed98d4b18718c6d22b3171faf5e9e9fc8a23"
_QSL_SOURCE_SHA256 = "4318e39b5e7c418b905874a87466247d5d21f13ff0919ce3b9258cf7f72324d0"
_QSL_MARKER = "_vspec_q35_draft_qsl_device_snapshot"
_QSL_CODE_QUALNAME = "apply_q35_draft_qsl_device_patch.<locals>.build"


class ShadowMismatch(RuntimeError):
    def __init__(self, field: str, reason: str):
        self.field = field
        super().__init__(f"Q35 shadow mismatch {field}: {reason}")


def _check_native_function(fn: Any) -> Callable[..., Any]:
    if not inspect.isfunction(fn):
        raise RuntimeError("Q35 shadow native updater is not a Python function")
    if fn.__module__ != _NATIVE_MODULE or fn.__qualname__ != _NATIVE_QUALNAME:
        raise RuntimeError("Q35 shadow updater identity differs from the pinned Ascend host method")
    if Path(fn.__code__.co_filename).resolve() != _UPSTREAM_PATH.resolve():
        raise RuntimeError("Q35 shadow updater code filename differs from pinned Ascend source")
    if tuple(inspect.signature(fn).parameters) != _UPDATER_PARAMETERS:
        raise RuntimeError("Q35 shadow updater signature changed")
    if hashlib.sha256(_UPSTREAM_PATH.read_bytes()).hexdigest() != _UPSTREAM_SHA256:
        raise RuntimeError("Q35 shadow Ascend source SHA-256 changed")
    return fn


def resolve_native_updater() -> Callable[..., Any]:
    """Accept direct native method or a proved compact wrapper closure only."""
    from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

    installed = AscendSpecDecodeBaseProposer.__dict__["attn_update_stack_num_spec_norm"]
    if getattr(installed, "__module__", None) == _NATIVE_MODULE:
        return _check_native_function(installed)
    if (getattr(installed, "__module__", None) != "vllm_hust_vspec.backends.draft"
            or getattr(installed, "__name__", None) != "compact_attn_update_stack_num_spec_norm"):
        raise RuntimeError("Q35 shadow cannot prove installed Base updater provenance")
    closure = inspect.getclosurevars(installed).nonlocals
    if "original_attn_update" not in closure:
        raise RuntimeError("Q35 shadow compact wrapper lacks native updater closure")
    return _check_native_function(closure["original_attn_update"])


def _clone_value(value: Any, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if value is None or isinstance(value, (str, bytes, int, float, bool, Enum, torch.dtype, torch.device)):
        return value
    if isinstance(value, list):
        return [_clone_value(item, f"{field}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, tuple):
        return tuple(_clone_value(item, f"{field}[{i}]") for i, item in enumerate(value))
    if isinstance(value, dict):
        return {key: _clone_value(item, f"{field}[{key}]") for key, item in value.items()}
    raise RuntimeError(f"Q35 shadow cannot isolate metadata field {field}: {type(value).__name__}")


def _clone_metadata(value: Any) -> Any:
    if not is_dataclass(value):
        raise RuntimeError("Q35 shadow common metadata is not a dataclass")
    names = {field.name for field in fields(value)}
    attributes = vars(value)
    if not names.issubset(attributes):
        raise RuntimeError("Q35 shadow common metadata has unset dataclass fields")
    cloned = copy.copy(value)
    # Include runtime-added attributes: copy.copy otherwise leaves their
    # mutable values aliased to live, and dataclass.fields would miss them.
    for name, item in attributes.items():
        setattr(cloned, name, _clone_value(item, name))
    return cloned


def _compare(expected: Any, actual: Any, field: str) -> None:
    if type(expected) is not type(actual):
        raise ShadowMismatch(field, f"type {type(expected).__name__} != {type(actual).__name__}")
    if isinstance(expected, torch.Tensor):
        if (expected.shape != actual.shape or expected.dtype != actual.dtype
                or expected.device != actual.device or not torch.equal(expected, actual)):
            raise ShadowMismatch(field, "tensor shape/dtype/device/content")
        return
    if isinstance(expected, np.ndarray):
        if expected.shape != actual.shape or expected.dtype != actual.dtype or not np.array_equal(expected, actual):
            raise ShadowMismatch(field, "NumPy shape/dtype/content")
        return
    if is_dataclass(expected):
        names = {item.name for item in fields(expected)}
        left, right = vars(expected), vars(actual)
        if not names.issubset(left) or not names.issubset(right) or left.keys() != right.keys():
            raise ShadowMismatch(field, "dataclass or runtime attribute set")
        for name in sorted(left):
            _compare(left[name], right[name], f"{field}.{name}")
        return
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise ShadowMismatch(field, "sequence length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{field}[{index}]")
        return
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ShadowMismatch(field, "mapping keys")
        for key in expected:
            _compare(expected[key], actual[key], f"{field}[{key}]")
        return
    if expected is None or isinstance(expected, (str, bytes, int, float, bool, Enum, torch.dtype, torch.device)):
        if expected != actual:
            raise ShadowMismatch(field, f"value {expected!r} != {actual!r}")
        return
    raise ShadowMismatch(field, f"unsupported metadata type {type(expected).__name__}")


class _ShadowGroup:
    def __init__(self, builder: Any):
        self._builder = builder
        self.kv_cache_spec = builder.kv_cache_spec

    def get_metadata_builder(self) -> Any:
        return self._builder


class _ShadowProposer:
    """Allowlisted dependencies of the pinned native, non-DCP updater."""

    def __init__(self, live: Any, builder_group: _ShadowGroup):
        self.method = live.method
        self.uses_mrope = live.uses_mrope
        self.has_gdn = live.has_gdn
        self.kernel_block_size = live.kernel_block_size
        self.block_size = live.block_size
        self.max_model_len = live.max_model_len
        self.use_compress = False
        self.sliding_window = None
        self.runner = SimpleNamespace(dcp_manager=None, sparse_kv_offload_enabled=False)
        self.arange = live.arange.clone()
        self.token_arange_np = live.token_arange_np.copy()
        self.slot_mapping_group = [tensor.clone() for tensor in live.slot_mapping_group]
        self.seq_lens_group = [tensor.clone() for tensor in live.seq_lens_group]
        self.query_start_loc_group = [tensor.clone() for tensor in live.query_start_loc_group]
        self.draft_attn_groups = [builder_group]

    def shallow_copy_metadata(self, metadata: Any) -> Any:
        return copy.copy(metadata)

    def _adjust_tensor(self, tensor: torch.Tensor, desired_size: int) -> torch.Tensor:
        pad_size = desired_size - tensor.shape[0]
        if pad_size > 0:
            pad = [0] * (2 * tensor.dim() - 1) + [pad_size]
            return F.pad(tensor, pad, mode="constant", value=0)
        return tensor[:desired_size]


@dataclass
class ShadowReference:
    gamma: int
    used: torch.Tensor
    proposer: _ShadowProposer
    results: list[tuple[Any, Any]]
    original_used: torch.Tensor
    original_old: Any
    frozen_old: Any
    original_inputs: dict[str, tuple[Any, Any]]
    group_addresses: tuple[tuple[int, ...], ...]


def _group_sets(proposer: Any) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    return (proposer.slot_mapping_group, proposer.seq_lens_group,
            proposer.query_start_loc_group)


def _addresses(groups: tuple[list[torch.Tensor], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(tensor.data_ptr() for tensor in group) for group in groups)


def _check_source_identity(old: Any, originals: dict[str, tuple[Any, Any]]) -> None:
    current = vars(old)
    for name, (source, _) in originals.items():
        if isinstance(source, (torch.Tensor, np.ndarray, list, tuple, dict)) and current.get(name) is not source:
            raise ShadowMismatch(f"source.{name}", "source object rebound")


def _check_native_builder_build(fn: Any) -> Callable[..., Any]:
    if (
        not inspect.isfunction(fn)
        or fn.__module__ != _NATIVE_BUILDER_MODULE
        or fn.__qualname__ != _NATIVE_BUILDER_QUALNAME
        or Path(fn.__code__.co_filename).resolve() != _NATIVE_BUILDER_PATH.resolve()
        or hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest()
        != _NATIVE_BUILDER_CODE_SHA256
    ):
        raise RuntimeError("Q35 shadow native attention builder provenance changed")
    return fn


def _check_builder_build(fn: Any) -> None:
    """Permit only the pinned native build or this source's QSL wrapper of it.

    ``functools.wraps`` makes the QSL wrapper's module and qualname look
    native, so those attributes alone cannot prove the installed method.
    The actual code object, globals, source, marker, ``__wrapped__``, and
    closed-over ``original_build`` must all agree before shadow construction.
    """
    if not inspect.isfunction(fn):
        raise RuntimeError("Q35 shadow attention builder build is not a Python function")
    if Path(fn.__code__.co_filename).resolve() == _NATIVE_BUILDER_PATH.resolve():
        _check_native_builder_build(fn)
        return

    from vllm_hust_vspec.backends import mtp_qsl_device as qsl
    from vllm.logger import logger as vllm_logger

    source = Path(__file__).resolve().with_name("mtp_qsl_device.py")
    if (
        Path(qsl.__file__).resolve() != source
        or hashlib.sha256(source.read_bytes()).hexdigest() != _QSL_SOURCE_SHA256
        or getattr(qsl, "_MARKER", None) != _QSL_MARKER
    ):
        raise RuntimeError("Q35 shadow QSL wrapper source changed")
    expected_codes = [
        obj for obj in qsl.apply_q35_draft_qsl_device_patch.__code__.co_consts
        if isinstance(obj, CodeType) and obj.co_qualname == _QSL_CODE_QUALNAME
    ]
    if (
        len(expected_codes) != 1
        or fn.__code__ is not expected_codes[0]
        or fn.__globals__ is not vars(qsl)
        or Path(fn.__code__.co_filename).resolve() != source
        or fn.__module__ != _NATIVE_BUILDER_MODULE
        or fn.__qualname__ != _NATIVE_BUILDER_QUALNAME
        or getattr(fn, _QSL_MARKER, None) is not True
        or fn.__code__.co_freevars != ("logger", "original_build")
        or fn.__closure__ is None
        or len(fn.__closure__) != 2
    ):
        raise RuntimeError("Q35 shadow attention builder QSL wrapper provenance changed")
    if fn.__closure__[0].cell_contents is not vllm_logger:
        raise RuntimeError("Q35 shadow QSL wrapper logger closure changed")
    original = fn.__closure__[1].cell_contents
    if getattr(fn, "__wrapped__", None) is not original:
        raise RuntimeError("Q35 shadow QSL wrapper native closure differs from __wrapped__")
    _check_native_builder_build(original)


def _new_builder(attn_group: Any) -> Any:
    from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder

    builder = attn_group.get_metadata_builder()
    if type(builder) is not AscendAttentionMetadataBuilder:
        raise RuntimeError("Q35 shadow builder type is not exact pinned full-attention builder")
    if AscendAttentionMetadataBuilder.__dict__["build"] is not builder.build.__func__:
        raise RuntimeError("Q35 shadow attention builder has an instance-level build override")
    _check_builder_build(builder.build.__func__)
    if (
        builder.build_for_drafting.__func__.__module__ != "vllm.v1.attention.backend"
        or builder.build_for_drafting.__func__.__qualname__ != "AttentionMetadataBuilder.build_for_drafting"
        or Path(builder.build_for_drafting.__func__.__code__.co_filename).resolve()
        != (_VLLM_ROOT / 'vllm/v1/attention/backend.py').resolve()
    ):
        raise RuntimeError("Q35 shadow attention builder method provenance changed")
    for filename, expected in _BUILDER_SOURCES.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Q35 shadow attention builder source changed: {filename}")
    shadow_builder = AscendAttentionMetadataBuilder(
        builder.kv_cache_spec, list(builder.layer_names), builder.vllm_config, builder.device
    )
    if shadow_builder is builder:
        raise RuntimeError("Q35 shadow attention builder aliases live builder")
    # AscendAttentionMetadataBuilder.__init__ obtains a singleton mask builder.
    # It must not share mutable cache fields with the live builder.
    shadow_builder.attn_mask_builder = _isolate_mask_builder(builder.attn_mask_builder)
    if shadow_builder.attn_mask_builder is builder.attn_mask_builder:
        raise RuntimeError("Q35 shadow mask builder aliases live singleton")
    return shadow_builder


def _isolate_mask_builder(mask: Any) -> Any:
    if (
        type(mask).__module__ != "vllm_ascend.attention.attention_mask"
        or type(mask).__name__ != "AttentionMaskBuilder"
    ):
        raise RuntimeError("Q35 shadow mask builder type changed")
    if set(vars(mask)) != {"attn_mask_cache", "_seq_len_cached", "device", "chunked_prefill_attn_mask"}:
        raise RuntimeError("Q35 shadow mask builder has unrecognized mutable fields")
    mask_source = (_ASCEND_ROOT / 'vllm_ascend/attention/attention_mask.py').resolve()
    for name in ("get_attention_mask", "get_attn_mask", "get_splitfuse_attn_mask"):
        method = getattr(mask, name).__func__
        if method.__module__ != "vllm_ascend.attention.attention_mask" or Path(
            method.__code__.co_filename
        ).resolve() != mask_source:
            raise RuntimeError("Q35 shadow mask builder method provenance changed")
    isolated_mask = copy.copy(mask)
    for name in ("attn_mask_cache", "chunked_prefill_attn_mask"):
        value = getattr(mask, name)
        if value is not None and not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Q35 shadow mask cache {name} has unexpected type")
        setattr(isolated_mask, name, None if value is None else value.clone())
        if value is not None and getattr(isolated_mask, name).untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise RuntimeError(f"Q35 shadow mask cache {name} aliases live cache")
    return isolated_mask


def prepare_shadow(
    live: Any, call: Any, gamma: int, *,
    updater: Callable[..., Any] | None = None,
    builder_factory: Callable[[Any], Any] | None = None,
) -> ShadowReference:
    """Compute all original host steps before the batch kernel; no live writes."""
    if gamma not in (4, 6) or call.draft_index != 1:
        raise RuntimeError("Q35 shadow must start at continuation step 1")
    if (call.ori_seq_len is not None or call.ori_seq_len_cpu is not None
            or call.slot_indices is not None or call.mtp_slot_mapping is not None):
        raise RuntimeError("Q35 shadow rejects DCP arguments")
    if live.method != "mtp" or not live.uses_mrope or live.use_compress or live.sliding_window is not None:
        raise RuntimeError("Q35 shadow only supports Q35 MTP non-compressed, non-windowed path")
    if getattr(live.runner, "dcp_manager", None) is not None or getattr(live.runner, "sparse_kv_offload_enabled", False):
        raise RuntimeError("Q35 shadow rejects DCP and sparse KV paths")
    updater = resolve_native_updater() if updater is None else updater
    builder_factory = _new_builder if builder_factory is None else builder_factory
    live_groups = _group_sets(live)
    original_inputs: dict[str, tuple[Any, Any]] = {}
    old = call.old_common_metadata
    if not is_dataclass(old):
        raise RuntimeError("Q35 shadow common metadata is not a dataclass")
    for name, source in vars(old).items():
        original_inputs[name] = (source, _clone_value(source, name))
    builder_group = _ShadowGroup(builder_factory(call.attn_group))
    shadow = _ShadowProposer(live, builder_group)
    shadow_used = call.used_update_positions.clone()
    initial_used = call.used_update_positions.clone()
    frozen_old = _clone_metadata(old)
    shadow_old = _clone_metadata(old)
    results: list[tuple[Any, Any]] = []
    for index in range(1, gamma):
        result = updater(
            shadow, index, shadow_old, call.batch_size, call.input_batch_size,
            shadow_used, call.aclgraph_runtime_mode, attn_group=builder_group,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("Q35 shadow native updater returned malformed result")
        results.append(result)
        shadow_old = result[0]
    # The shadow native updater must not have written through any live source.
    _compare(frozen_old, old, "source")
    _check_source_identity(old, original_inputs)
    _compare(initial_used, call.used_update_positions, "source.used_before_launch")
    return ShadowReference(
        gamma, shadow_used, shadow, results, initial_used, old, frozen_old,
        original_inputs,
        _addresses(live_groups),
    )


def compare_shadow(
    live: Any, live_used: torch.Tensor, actual_results: list[tuple[Any, Any]],
    reference: ShadowReference,
) -> None:
    """Compare before draft graph replay; any unhandled field is a hard error."""
    if len(actual_results) != reference.gamma - 1:
        raise ShadowMismatch("results", "missing continuation result")
    live_groups = _group_sets(live)
    if _addresses(live_groups) != reference.group_addresses:
        raise ShadowMismatch("group_addresses", "graph-stable destination address changed")
    shadow_groups = _group_sets(reference.proposer)
    for kind, (left_group, right_group) in enumerate(zip(shadow_groups, live_groups, strict=True)):
        if len(left_group) != len(right_group):
            raise ShadowMismatch(f"groups[{kind}]", "group count")
        for index, (left, right) in enumerate(zip(left_group, right_group, strict=True)):
            if left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr():
                raise ShadowMismatch(f"groups[{kind}][{index}]", "shadow aliases live")
            _compare(left, right, f"groups[{kind}][{index}]")
    _compare(reference.used, live_used, "used")
    position_addresses = []
    for index, (expected, actual) in enumerate(zip(reference.results, actual_results, strict=True), 1):
        _compare(expected[0], actual[0], f"common[{index}]")
        _compare(expected[1], actual[1], f"attention[{index}]")
        position_addresses.append(actual[0].positions.untyped_storage().data_ptr())
    if len(set(position_addresses)) != len(position_addresses):
        raise ShadowMismatch("positions", "step buffers alias")
    live_source_addresses = {
        ref.untyped_storage().data_ptr()
        for ref, _ in reference.original_inputs.values()
        if isinstance(ref, torch.Tensor)
    }
    live_group_addresses = {address for group in reference.group_addresses for address in group}
    if set(position_addresses) & (live_source_addresses | live_group_addresses):
        raise ShadowMismatch("positions", "step buffer aliases input or persistent group")
    _compare(reference.frozen_old, reference.original_old, "source")
    _check_source_identity(reference.original_old, reference.original_inputs)
