"""Validate native FIA host batching on the two allocated NPUs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .backends.experimental_paths import current_npu_allocation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(directory: Path) -> None:
    directory = directory.resolve()
    output = directory / 'primitive-native-audit-r1.json'
    if output.exists():
        raise ValueError('Proof already exists; use a new build directory')
    for name in ('micro.py', 'guard_controls.py', 'adapter.cpp', 'CMakeLists.txt', 'build-r1/lib/vspec_fia_host.so'):
        if not (directory / name).is_file():
            raise FileNotFoundError(f'Built FIA component is missing {name}')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('Both allocated NPUs must be idle')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', VLLM_PLUGINS='',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR', 'TMPDIR'):
        path = directory / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    ranks = []
    for rank in (0, 1):
        result = directory / f'fia-host-d{rank}-r1.json'
        with (directory / f'fia-host-d{rank}-r1.log').open('x') as log:
            subprocess.run([sys.executable, '-B', str(directory / 'micro.py'), '--device', str(rank), '--output', str(result)],
                           cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = json.loads(result.read_text())
        if not row['passed'] or row['device'] != rank or row['comparisons'] != 1080 or row['compiled_guard_controls'] != 55:
            raise RuntimeError('FIA numerical validation incomplete')
        if {(r['requests'], r['role'], r['gamma']) for r in row['records']} != {
                (n, role, gamma) for n in (1, 2, 3, 14) for role in ('target', 'draft') for gamma in (4, 6)}:
            raise RuntimeError('FIA validation matrix incomplete')
        for record in row['records']:
            count = 10 if record['role'] == 'target' else record['gamma']
            widths = [record['gamma'] + 1] * 10 if record['role'] == 'target' else [record['gamma'] + 1] + [1] * (count - 1)
            if record['native_groups'] != count or record['widths'] != widths or record['output_comparisons'] != 9 * count:
                raise RuntimeError('FIA comparison scope incomplete')
            for flag in ('all_8_replays_use_changed_exact_host_lengths', 'per_task_distinct_KV_and_replaced_block_table_tensors',
                         'capture_query_lengths_full_outer_then_runtime_exact_live_prefix', 'original_per_row_padding_after_real_requests_preserved',
                         'unused_padding_outputs_NOT_counted_as_semantic_rows'):
                if not record[flag]:
                    raise RuntimeError('FIA graph comparison failed')
        if any(digest(Path(name)) != value for name, value in row['sha256'].items()):
            raise RuntimeError('FIA component input changed during validation')
        ranks.append(row)
    if current_npu_allocation() != physical:
        raise RuntimeError('Device allocation changed during validation')
    hashes = {name: value for row in ranks for name, value in row['sha256'].items()}
    hashes.update({str(directory / f'fia-host-d{rank}-r1.json'): digest(directory / f'fia-host-d{rank}-r1.json') for rank in (0, 1)})
    proof = {'passed': True, 'scope': 'target-draft', 'physical_npus': physical, 'numerical_comparisons': 2160,
             'all_consumed_output_rows_finite_and_bitwise_exact': True,
             'original_per_group_update_and_ExternalEvent_order_preserved': True,
             'component_only_NOT_model_speed_quality_or_goal_admission': True, 'ranks': ranks, 'sha256': hashes}
    with output.open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.directory)
    print('Component proof passed; end-to-end model validation is still required.')


if __name__ == '__main__':
    main()
