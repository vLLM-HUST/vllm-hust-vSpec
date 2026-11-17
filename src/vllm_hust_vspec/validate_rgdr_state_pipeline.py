"""Check and validate the standalone RGDR state pipeline operator.

``--cpu-check`` checks a built adapter without opening an NPU. The default
mode compares both NPU ranks against the installed native recurrent operator,
including every state slot through eager execution and captured-graph replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .backends.experimental_paths import (
    configure_custom_opp,
    current_npu_allocation,
    require_current_npu_proof,
)


SOURCE_NAMES = (
    'micro.py', 'adapter.cpp', 'kernel.cpp', 'rgdr_full128.h',
    'recurrent_gated_delta_rule_tiling_data.h', 'CMakeLists.txt', 'CANN-LICENSE',
)
LIBRARY_NAMES = ('libvspec_rgdr_adapter.so', 'libvspec_rgdr_kernel.so')
RUN_NAME = 'rgdr-state-write-pipeline-v2-r1'
EXPECTED_COPIED_SOURCE_SHA256 = {
    'micro.py': 'b0a20dd280065b32110a862ef633236ea773fe922d3e2178d9db1f96c568e9d1',
    'adapter.cpp': '7afd254d93eb488b9b1c0ec3914a258e8e671605c0057b2494247beac001d0cd',
    'kernel.cpp': 'f04c8542d77547d637af14df5f5ca6df1bed279c7c0e6dc250b50855ec41635b',
    'rgdr_full128.h': '257074f2bf95bf733c80609bedad481bcffd7e8e5e5f6aee180751eeb16bfa29',
    'recurrent_gated_delta_rule_tiling_data.h':
        '578015f07642a2260f6689a83c4ec44cc4a289ab3c041e15a038a5654c2b49de',
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_build(component_dir: Path) -> dict[str, str]:
    """Verify source identity and the loadable kernel linkage without an NPU."""
    root = component_dir.resolve()
    paths = [root / name for name in SOURCE_NAMES]
    paths.extend(root / 'build-r1/lib' / name for name in LIBRARY_NAMES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('RGDR state pipeline source or library missing: ' + ', '.join(missing))
    hashes = {str(path): _sha(path) for path in paths}
    for name, expected in EXPECTED_COPIED_SOURCE_SHA256.items():
        if hashes[str(root / name)] != expected:
            raise RuntimeError(f'Packaged RGDR source differs from the validated prototype: {name}')
    adapter = root / 'build-r1/lib/libvspec_rgdr_adapter.so'
    kernel = root / 'build-r1/lib/libvspec_rgdr_kernel.so'
    for path in (adapter, kernel):
        elf = subprocess.check_output(['readelf', '-h', str(path)], text=True)
        if 'ELF64' not in elf or 'AArch64' not in elf or 'DYN (Shared object file)' not in elf:
            raise RuntimeError(f'Unexpected RGDR library format: {path}')
    dynamic = subprocess.check_output(['readelf', '-d', str(adapter)], text=True)
    if 'libvspec_rgdr_kernel.so' not in dynamic:
        raise RuntimeError('RGDR adapter is not linked to the bundled kernel')
    return hashes


def validate(component_dir: Path, backend_package: Path) -> Path:
    """Produce a two-rank numerical proof for this physical NPU allocation."""
    root, backend = component_dir.resolve(), backend_package.resolve()
    hashes = check_build(root)
    spec = importlib.util.find_spec('vllm_ascend')
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(backend):
        raise RuntimeError('Active vllm_ascend is not the explicitly selected backend')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count(
            'No running processes found') != 2:
        raise RuntimeError('Both allocated NPUs must be idle before numerical validation')
    runs = root / 'runs'
    if runs.exists():
        raise FileExistsError('Component output exists; use a fresh build directory')
    os.umask(0o077)
    runs.mkdir(mode=0o700)
    env = dict(os.environ)
    configure_custom_opp(env, backend)
    lib_dir = str(root / 'build-r1/lib')
    env['LD_LIBRARY_PATH'] = lib_dir + ':' + env.get('LD_LIBRARY_PATH', '')
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH',
                'ACL_OP_COMPILER_CACHE_DIR'):
        path = root / 'validation-cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    tmp = root / 'validation-tmp'
    tmp.mkdir(mode=0o700)
    env.update(TMPDIR=str(tmp), VLLM_PLUGINS='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('PYTHONPATH', None)
    output = runs / RUN_NAME
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=2', str(root / 'micro.py'),
               '--output', str(output.with_suffix('.json'))]
    with output.with_suffix('.log').open('x') as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    expected = {(width, seed) for width in (5, 7) for seed in range(24)}
    variants = {'native', 'base128', 'serial96', 'pipe96', 'serial80', 'pipe80',
                'serial64', 'pipe64'}
    ranks = []
    for index in (0, 1):
        result = output.with_name(output.name + f'-rank{index}.json')
        rank = json.loads(result.read_text())
        rows = rank['records']
        if (not rank['completed'] or rank['rank'] != index or len(rows) != 48
                or {(row['width'], row['seed']) for row in rows} != expected):
            raise RuntimeError('RGDR state pipeline validation matrix is incomplete')
        if {row['route'] for row in rows} != {'uniform', 'uneven', 'padded_zero'}:
            raise RuntimeError('RGDR metadata routes were not all checked')
        for row in rows:
            if not (row['all_eager_and_eight_graph_outputs_and_entire_state_bits_exact']
                    and row['native_eager_and_graph_self_reference_passed']
                    and row['unused_NaN_payload_slot127_preserved']):
                raise RuntimeError('RGDR numerical or full-state comparison failed')
            timings = row['median_us']
            if set(timings) != variants or not all(
                    math.isfinite(value) and value > 0 for value in timings.values()):
                raise RuntimeError('RGDR timing record is incomplete')
        for filename, digest in rank['sha256'].items():
            if _sha(Path(filename)) != digest:
                raise RuntimeError('RGDR component source changed during numerical validation')
        ranks.append(rank)
    if any(_sha(Path(filename)) != digest for filename, digest in hashes.items()):
        raise RuntimeError('RGDR component source changed during numerical validation')
    proof.update(completed=True, passed=True, exact=True, ranks=ranks, sha256=hashes,
                 cases_per_rank=48, component_only_no_model_or_speed_or_quality_admission=True)
    target = runs / (RUN_NAME + '.primitive-audit.json')
    with target.open('x') as stream:
        json.dump(proof, stream, indent=2)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-dir', required=True, type=Path)
    parser.add_argument('--backend-package', type=Path,
                        help='Installed vllm_ascend site-packages directory for numerical validation')
    parser.add_argument('--cpu-check', action='store_true',
                        help='Check the build and copied source identity without opening an NPU')
    args = parser.parse_args()
    if args.cpu_check:
        check_build(args.component_dir)
        print('RGDR state pipeline build and copied source checks passed; NPU unvalidated.')
        return
    if args.backend_package is None:
        parser.error('--backend-package is required for numerical validation')
    proof = validate(args.component_dir, args.backend_package)
    print(f'RGDR state pipeline component proof passed: {proof}')


if __name__ == '__main__':
    main()
