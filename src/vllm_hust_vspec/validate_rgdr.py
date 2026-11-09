"""Validate freshly built reference-six or gamma-eight recurrent operators."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .backends.experimental_paths import current_npu_allocation, require_current_npu_proof


def validate(component: str, root: Path, backend_package: Path) -> None:
    configs = {
        'gamma6-reference': ('vspec_rgdr_tail6', 'gamma6-tail-c46-r1', 7),
        'gamma8': ('vspec_rgdr_edge8', 'gamma8-allb-c46-r1', 9),
    }
    if component not in configs:
        raise ValueError('Unsupported recurrent operator')
    lib, prefix, width = configs[component]
    root, backend_package = root.resolve(), backend_package.resolve()
    spec = importlib.util.find_spec('vllm_ascend')
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(backend_package):
        raise RuntimeError('Active vllm_ascend is not the explicitly selected new backend')
    sources = [root / name for name in ('micro.py', 'kernel.cpp', 'adapter.cpp', 'CMakeLists.txt',
               'rgdr_full128.h', 'recurrent_gated_delta_rule_tiling_data.h')]
    sources.extend(root / 'build-r1/lib' / f'lib{lib}_{name}.so' for name in ('adapter', 'kernel'))
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError('Built component or validation source is missing')
    output = root / 'runs'
    if output.exists():
        raise FileExistsError('Component output exists; use a fresh build directory')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    os.umask(0o077)
    output.mkdir(mode=0o700)
    env = dict(os.environ)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = root / 'validation-cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    temp = root / 'validation-tmp'
    temp.mkdir(mode=0o700)
    env.update(TMPDIR=str(temp), VLLM_PLUGINS='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('PYTHONPATH', None)
    name = output / prefix
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=2', str(root / 'micro.py'), '--output', str(name.with_suffix('.json'))]
    with name.with_suffix('.log').open('x') as log:
        subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    ranks = [json.loads(name.with_name(name.name + f'-rank{rank}.json').read_text()) for rank in (0, 1)]
    expected = {(n, bank, seed) for n in range(1, 17) for bank in (192, 647) for seed in range(6)}
    required = ('independent_native_eager_graph_reference', 'eight_eager_graph_revisions_bits_exact',
                'captured_graph_dynamic_lengths_and_accepted_covered',
                'full_FP32_state_including_untouched_NaN_payload_and_signed_zero_exact',
                'complete_inputs_bits_readonly', 'entire_folded_counter_array_exact')
    for index, rank in enumerate(ranks):
        if not rank['completed'] or rank['rank'] != index or len(rank['records']) != 192:
            raise RuntimeError('Incomplete recurrent component results')
        if {(row['requests'], row['bank'], row['seed']) for row in rank['records']} != expected:
            raise RuntimeError('Recurrent component test matrix is incomplete')
        if not all(row['width'] == width and all(row[key] for key in required) for row in rank['records']):
            raise RuntimeError('Recurrent component numerical validation failed')
        for filename, digest in rank['sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError('A recurrent test input changed during validation')
    if any(hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest for filename, digest in hashes.items()):
        raise RuntimeError('A recurrent source changed during validation')
    proof.update(passed=True, exact=True, ranks=ranks, sha256=hashes,
                 component_only_NOT_model_quality_or_end_to_end_speed_admission=True)
    with name.with_suffix('.primitive-audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('gamma6-reference', 'gamma8'), required=True)
    parser.add_argument('--component-dir', required=True, type=Path)
    parser.add_argument('--backend-package', required=True, type=Path)
    args = parser.parse_args()
    validate(args.component, args.component_dir, args.backend_package)
    print('Recurrent component proof passed; this is not an end-to-end result.')


if __name__ == '__main__':
    main()
