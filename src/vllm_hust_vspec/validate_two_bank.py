"""Validate the two-bank embedding component against a local checkpoint on two NPUs."""
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

from .backends.experimental_paths import current_npu_allocation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(model: Path, destination: Path) -> None:
    model, destination = model.resolve(), destination.resolve()
    if not (model / 'model.safetensors.index.json').is_file():
        raise FileNotFoundError('Model checkpoint index not found')
    if destination.exists() or destination.is_relative_to(model):
        raise ValueError('Destination must be a new private directory outside the model')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('Both allocated NPUs must be idle')
    source = Path(str(importlib.resources.files('vllm_hust_vspec') / '_vendor/operators/two_bank_embedding'))
    destination.mkdir(parents=True, mode=0o700)
    for name in ('kernel.py', 'opaque.py', 'micro.py', 'opaque_smoke.py'):
        shutil.copy2(source / name, destination / name)
    env = dict(os.environ, VSPEC_MODEL_DIR=str(model), PYTHONDONTWRITEBYTECODE='1',
               VLLM_PLUGINS='', OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('ASCEND_RT_VISIBLE_DEVICES', None)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR', 'TMPDIR'):
        path = destination / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)

    def run(script: str, logical: int) -> None:
        with (destination / f'{script}-{logical}.log').open('x') as log:
            subprocess.run([sys.executable, '-B', str(destination / script), '--logical', str(logical)],
                           cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def read_proofs(prefix: str) -> list[dict]:
        result = []
        for logical in (0, 1):
            row = json.loads((destination / f'{prefix}-card{logical + 1}-r1.json').read_text())
            if not row['passed'] or row['physical_npus'] != [physical[logical]]:
                raise RuntimeError('Component proof did not pass on the allocated device')
            if any(digest(Path(name)) != value for name, value in row['sha256'].items()):
                raise RuntimeError('A component input changed during validation')
            result.append(row)
        if current_npu_allocation() != physical:
            raise RuntimeError('Device allocation changed during validation')
        return result

    for rank in (0, 1):
        run('micro.py', rank)
    ranks = read_proofs('split-embedding')
    for rank in ranks:
        if {r['block'] for r in rank['all_vocab_checks'] if r['all248320_vocabulary_rows_bitwise_exact']} != {512, 1024, 2048}:
            raise RuntimeError('Full-vocabulary validation is incomplete')
        if {r['rows'] for r in rank['records'] if r['eager_and8_graph_revisions_all_consumed_rows_bitwise_exact']} != {1, 2, 3, 7, 14, 16, 21, 35, 70, 98, 208, 8192}:
            raise RuntimeError('Embedding graph validation is incomplete')
    if ranks[0]['embedding_tensor_sha256'] != ranks[1]['embedding_tensor_sha256']:
        raise RuntimeError('Ranks read different checkpoint weights')
    hashes = {str(destination / name): digest(destination / name) for name in ('kernel.py', 'opaque.py', 'micro.py', 'opaque_smoke.py')}
    primitive = {'passed': True, 'physical_npus': physical, 'ranks': ranks, 'sha256': hashes,
                 'component_only_NOT_model_speed_quality_or_goal_admission': True}
    with (destination / 'split-embedding-r1-primitive-audit.json').open('x') as stream:
        json.dump(primitive, stream, indent=2)
    for rank in (0, 1):
        run('opaque_smoke.py', rank)
    ranks = read_proofs('opaque')
    for rank in ranks:
        if {r['rows'] for r in rank['records'] if r['compiled_NPU_output_bits_exact'] and r['captured_graph_output_bits_exact'] and r['mutable_counter_delta_for5_replays'] == 5} != {14, 70, 98}:
            raise RuntimeError('Opaque graph validation is incomplete')
    hashes[str(destination / 'split-embedding-r1-primitive-audit.json')] = digest(destination / 'split-embedding-r1-primitive-audit.json')
    with (destination / 'opaque-r1-audit.json').open('x') as stream:
        json.dump({'passed': True, 'physical_npus': physical, 'ranks': ranks, 'sha256': hashes,
                   'component_only_NOT_model_speed_quality_or_goal_admission': True}, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.model, args.destination)
    print('Component proof passed; end-to-end model validation is still required.')


if __name__ == '__main__':
    main()
