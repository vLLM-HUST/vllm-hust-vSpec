"""Validate the experimental verified-prefix clip component on two devices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .backends.experimental_paths import configure_custom_opp, current_npu_allocation, require_current_npu_proof


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(backend: Path, destination: Path) -> None:
    backend, destination = backend.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(backend):
        raise ValueError('Destination must be new and outside the backend')
    spec = importlib.util.find_spec('vllm_ascend')
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(backend):
        raise RuntimeError('Active vllm_ascend is not the explicitly selected backend')
    ascend = backend / 'vllm_ascend'
    native = [ascend / item for item in ('spec_decode/llm_base_proposer.py', 'ops/triton/reject_sample.py',
              'ops/triton/triton_utils.py', 'device/device_op.py')]
    if not all(path.is_file() for path in native):
        raise FileNotFoundError('Selected backend reference sources are missing')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    source = Path(str(importlib.resources.files('vllm_hust_vspec') / '_vendor/operators/prefix_clip'))
    destination.mkdir(parents=True, mode=0o700)
    names = ('clip_kernel.py', 'micro.py', 'native_count_tail.py', 'reference.py')
    for name in names:
        shutil.copy2(source / name, destination / name)
    paths = [destination / name for name in names] + native
    hashes = {str(path): sha(path) for path in paths}
    env = dict(os.environ)
    configure_custom_opp(env, backend)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = destination / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    tmp = destination / 'tmp'
    tmp.mkdir(mode=0o700)
    env.update(TMPDIR=str(tmp), PYTHONPATH=str(destination), VLLM_PLUGINS='',
               VSPEC_ASCEND_SOURCE_ROOT=str(backend), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    required = ('nine_revisions_including_all_rejection_positions_and_all_accept',
                'independent_native_wide_and_narrow_eager_self_graph',
                'all_output_ID_bits_and_native_count_next_ID_bits_exact',
                'original_wide_output_and_all_inputs_readonly', 'complete_in_kernel_folded_witness_exact',
                'asynchronous_eager_chain_32calls_without_intermediate_sync_verified',
                'native_reference_CU_guarded_backing_unselected_offset_minus1_read_explicit',
                'no_attention_KV_SSM_or_existing_count_tensor_access')
    expected = {(n, width, width - 2, discarded) for n in range(1, 17) for width in (4, 6, 8)
                for discarded in (0, 1)}
    ranks = []
    for rank in (0, 1):
        output = destination / f'rank{rank}.json'
        cmd = [sys.executable, '-B', str(destination / 'micro.py'), '--device', str(rank), '--output', str(output)]
        with (destination / f'rank{rank}.log').open('x') as log:
            subprocess.run(cmd, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(output.read_text())
        rows = data['records']
        if not data['completed'] or data['device'] != rank or len(rows) != 96:
            raise RuntimeError('Incomplete prefix clip component results')
        if {(r['requests'], r['old'], r['new'], r['discarded_requests']) for r in rows} != expected:
            raise RuntimeError('Prefix clip component test matrix is incomplete')
        if not all(all(row[key] for key in required) for row in rows):
            raise RuntimeError('Prefix clip numerical checks did not pass')
        for filename, digest in data['sha256'].items():
            if sha(Path(filename)) != digest:
                raise RuntimeError('A prefix clip test source changed')
        ranks.append(data)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    if any(sha(Path(filename)) != digest for filename, digest in hashes.items()):
        raise RuntimeError('A prefix clip source changed during validation')
    proof.update(passed=True, ranks=ranks, sha256=hashes,
                 component_only_NOT_live_async_bridge_state_or_quality_or_speed_admission=True)
    with (destination / 'prefix-bridge-component-audit-r1.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.backend, args.destination)
    print('Prefix clip component proof passed; this is not a model or speed result.')


if __name__ == '__main__':
    main()
