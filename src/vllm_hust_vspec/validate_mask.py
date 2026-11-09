"""Run the experimental post-top8 mask's two-device local component proof."""
from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .backends.experimental_paths import current_npu_allocation, require_current_npu_proof


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(backend: Path, destination: Path) -> None:
    backend, destination = backend.resolve(), destination.resolve()
    old = backend / 'vllm_ascend/ops/triton/q35_expert_mask.py'
    if not old.is_file():
        raise FileNotFoundError('Prepared Ascend backend reference kernel not found')
    if destination.exists() or destination.is_relative_to(backend):
        raise ValueError('Destination must be a new private directory outside the backend')
    physical = current_npu_allocation()
    info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15)
    if info.count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    source = Path(str(importlib.resources.files('vllm_hust_vspec') /
                      '_vendor/operators/gamma6_mask'))
    destination.mkdir(mode=0o700, parents=True)
    for name in ('micro.py', 'kernel.py'):
        shutil.copy2(source / name, destination / name)
    inputs = (destination / 'micro.py', destination / 'kernel.py', old)
    hashes = {str(path): sha(path) for path in inputs}
    env = dict(os.environ)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT',
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME',
                'TORCH_EXTENSIONS_DIR', 'HF_HOME', 'ASCEND_WORK_PATH',
                'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = destination / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    tmp = destination / 'tmp'
    tmp.mkdir(mode=0o700)
    env.update(TMPDIR=str(tmp), PYTHONPATH=str(destination), VLLM_PLUGINS='',
               VSPEC_ASCEND_SOURCE_ROOT=str(backend), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=2', str(destination / 'micro.py'), '--output', str(destination / 'result.json')]
    with (destination / 'run.log').open('x') as log:
        subprocess.run(cmd, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    ranks = [json.loads((destination / f'result-rank{rank}.json').read_text()) for rank in (0, 1)]
    required = ('independent_native_eager_self_graph_bits_exact',
                'eight_dynamic_input_eager_graph_revisions_bits_exact',
                'all_raw_output_bits_NaN_payload_signed_zero_infinity_subnormal_covered',
                'complete_input_storage_including_canaries_readonly',
                'all17_counter_cells_exact', 'router_scores_top8_and_weight_values_unchanged')
    expected = {(n, i, w) for n in range(1, 17) for i in ('torch.int32', 'torch.int64')
                for w in ('torch.bfloat16', 'torch.float32')}
    for index, rank in enumerate(ranks):
        if not rank['passed'] or rank['rank'] != index or len(rank['records']) != 64:
            raise RuntimeError('Incomplete component results')
        if {(r['requests'], r['id_dtype'], r['weight_dtype']) for r in rank['records']} != expected:
            raise RuntimeError('Component result matrix is incomplete')
        for row in rank['records']:
            if not all(row[key] for key in required):
                raise RuntimeError('Component numerical validation failed')
            if row['median_us']['native'] / row['median_us']['fused_folded_count'] < 1.01:
                raise RuntimeError('Component does not meet the recorded minimum speed gate')
        for name, digest in rank['sha256'].items():
            if sha(Path(name)) != digest:
                raise RuntimeError('A component input changed during validation')
    if any(sha(Path(name)) != digest for name, digest in hashes.items()):
        raise RuntimeError('A component input changed during validation')
    proof.update(passed=True, exact=True, ranks=ranks, sha256=hashes,
                 component_only_NOT_model_speed_quality_or_goal_admission=True,
                 only_post_native_top8_consumer_no_router_change=True)
    with (destination / 'primitive-audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.backend, args.destination)
    print('Component proof passed; this does not establish model correctness or speed.')


if __name__ == '__main__':
    main()
