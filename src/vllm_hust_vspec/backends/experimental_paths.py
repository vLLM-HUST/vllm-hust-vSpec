"""Locations of locally provisioned validation artifacts for experimental hooks.

These hooks remain fail-closed: configuring a directory does not bypass any of
their original source, numerical or device validation checks.
"""
import os
import re
import subprocess
from pathlib import Path


def artifact_dir(variable: str) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else Path('/__vspec_artifacts_not_configured__') / variable


def configure_custom_opp(env: dict[str, str], backend_package: Path) -> None:
    """Bind child processes to the selected installed backend's own custom ops."""
    vendor = (backend_package.resolve() / 'vllm_ascend/_cann_ops_custom/vendors/custom_transformer').resolve()
    library = vendor / 'op_api/lib/libcust_opapi.so'
    if not library.is_file():
        raise FileNotFoundError(f'Installed backend custom operator library missing: {library}')
    env['ASCEND_CUSTOM_OPP_PATH'] = str(vendor)
    previous = env.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = str(library.parent) + (':' + previous if previous else '')


def current_npu_allocation() -> list[int]:
    """Resolve the current supported allocation; never trust a device ID override.

    The experimental two-rank harness requires both visible logical devices,
    in logical order, to be 910B2s. Restricted or ambiguous mappings fail closed.
    """
    if os.environ.get('ASCEND_VISIBLE_DEVICES'):
        raise RuntimeError('ASCEND_VISIBLE_DEVICES prevents validating the allocation unambiguously')
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if visible and visible != '0,1':
        raise RuntimeError('ASCEND_RT_VISIBLE_DEVICES prevents validating the allocation unambiguously')
    output = subprocess.check_output(['npu-smi', 'info', '-m'], text=True, timeout=15)
    devices = re.findall(r'^\s*(\d+)\s+0\s+(\d+)\s+Ascend 910B2\s*$', output, re.MULTILINE)
    if len(devices) != 2 or {logical for _, logical in devices} != {'0', '1'}:
        raise RuntimeError('Expected exactly two visible Ascend 910B2 logical devices')
    current = [int(physical) for physical, _ in sorted(devices, key=lambda pair: int(pair[1]))]
    if len(set(current)) != 2:
        raise RuntimeError('Expected two distinct physical NPU devices')
    return current


def require_current_npu_proof(proof: dict) -> None:
    """Reject proofs from a different allocation."""
    if proof.get('physical_npus') != current_npu_allocation():
        raise RuntimeError('Validation proof does not match the current physical NPU allocation')
