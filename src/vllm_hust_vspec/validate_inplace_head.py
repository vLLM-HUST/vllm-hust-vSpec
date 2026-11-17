"""Validate the private draft-head layout against actual checkpoint weights."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .backends.experimental_paths import current_npu_allocation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def short_socket_tmpdir():
    """Keep multiprocessing manager sockets below the AF_UNIX path limit."""
    with tempfile.TemporaryDirectory(prefix='vspec-head-', dir='/tmp') as path:
        yield Path(path)


def validate(model: Path, destination: Path) -> None:
    model, destination = model.resolve(), destination.resolve()
    if not (model / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError('Model checkpoint index not found')
    if destination.exists() or destination.is_relative_to(model):
        raise ValueError('Destination must be a new private directory outside the model')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('Both allocated NPUs must be idle')
    source = Path(str(importlib.resources.files('vllm_hust_vspec') / '_vendor/operators/inplace_head_nz'))
    destination.mkdir(parents=True, mode=0o700)
    for name in ('kernel.py', 'micro.py', 'real_weight.py'):
        shutil.copy2(source / name, destination / name)
    inputs = [destination / name for name in ('kernel.py', 'micro.py', 'real_weight.py')]
    head = Path(str(importlib.resources.files('vllm_hust_vspec') / 'backends/mtp_active_head.py'))
    inputs.extend([head, model / 'model.safetensors.index.json'])
    hashes = {str(path): digest(path) for path in inputs}
    env = dict(os.environ, VSPEC_MODEL_DIR=str(model), PYTHONDONTWRITEBYTECODE='1',
               VLLM_PLUGINS='', OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = destination / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    out = destination / 'real-head-runtime-true-r1.json'
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
           str(destination / 'micro.py'), '--output', str(out)]
    with short_socket_tmpdir() as socket_tmp:
        env['TMPDIR'] = str(socket_tmp)
        with (destination / 'run.log').open('x') as log:
            subprocess.run(cmd, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if current_npu_allocation() != physical:
        raise RuntimeError('Device allocation changed during validation')
    ranks = [json.loads((destination / f'real-head-runtime-true-r1-rank{rank}.json').read_text()) for rank in (0, 1)]
    shapes = {'draft': set(range(1, 17)),
              'target': set(range(1, 17)) | {n * q for n in range(1, 17) for q in (5, 7)} | {208}}
    for rank, value in enumerate(ranks):
        if not value['completed'] or value['rank'] != rank or not value['real_checkpoint_weights']:
            raise RuntimeError('Incomplete checkpoint weight proof')
        if value['provenance']['rank'] != rank or not value['provenance']['only_public_weight_read_no_shared_writes']:
            raise RuntimeError('Unexpected checkpoint provenance')
        for role, expected in shapes.items():
            rows = [row for row in value['records'] if row['role'] == role]
            if len(rows) != len(expected) or {row['rows'] for row in rows} != expected:
                raise RuntimeError('Head validation matrix incomplete')
            for row in rows:
                if not row['actual_weight_bits_unchanged'] or len(row['checks']) != 8:
                    raise RuntimeError('Head weight or graph check incomplete')
                if not all(check['arms']['native']['exact'] for check in row['checks']):
                    raise RuntimeError('Native graph mismatch')
                if row['exact'] != all(check['arms']['candidate']['exact'] for check in row['checks']):
                    raise RuntimeError('Candidate result inconsistent')
        if any(digest(Path(path)) != value for path, value in value['sha256'].items()):
            raise RuntimeError('Component source changed during validation')
    eligible = {role: sorted(n for n in expected if all(next(row for row in rank['records']
                if row['role'] == role and row['rows'] == n)['exact'] for rank in ranks))
                for role, expected in shapes.items()}
    if eligible['draft'] != list(range(1, 17)):
        raise RuntimeError('Draft head does not pass all required shapes on both devices')
    if any(digest(Path(path)) != value for path, value in hashes.items()):
        raise RuntimeError('Component input changed during validation')
    proof = {'completed': True, 'physical_npus': physical, 'eligible_shapes': eligible, 'ranks': ranks,
             'actual_public_checkpoint_weights': True, 'sha256': hashes,
             'component_only_NOT_model_speed_quality_or_goal_admission': True}
    with (destination / 'real-head-runtime-true-r1-primitive-audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.model, args.destination)
    print('Component proof passed; end-to-end model validation is still required.')


if __name__ == '__main__':
    main()
