"""Validate the freshly built experimental Ascend tail convolution on two devices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .backends.experimental_paths import configure_custom_opp, current_npu_allocation, require_current_npu_proof


def validate(root: Path, backend_package: Path) -> None:
    root, backend_package = root.resolve(), backend_package.resolve()
    spec = importlib.util.find_spec('vllm_ascend')
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(backend_package):
        raise RuntimeError('Active vllm_ascend is not the explicitly selected new backend')
    paths = [root / 'micro.py', root / 'kernel.cpp', root / 'adapter.cpp', root / 'CMakeLists.txt',
             root / 'native-kernel/causal_conv1d.h',
             root / 'build-r1/lib/libvspec_g6_tail_v2_adapter.so',
             root / 'build-r1/lib/libvspec_g6_tail_v2_kernel.so']
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Built component or its validation source is missing')
    paths.extend(sorted((root / 'native-kernel').rglob('*.h')))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = root / 'runs'
    if output.exists():
        raise FileExistsError('Component test output already exists; use a fresh build directory')
    physical = current_npu_allocation()
    info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15)
    if info.count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    os.umask(0o077)
    output.mkdir(mode=0o700)
    env = dict(os.environ)
    configure_custom_opp(env, backend_package)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT',
                'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME',
                'TORCH_EXTENSIONS_DIR', 'HF_HOME', 'ASCEND_WORK_PATH',
                'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = root / 'validation-cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    temp = root / 'validation-tmp'
    temp.mkdir(mode=0o700)
    env.update(TMPDIR=str(temp), VLLM_PLUGINS='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('PYTHONPATH', None)
    name = output / 'gamma6-tail-strides-c46-r1'
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
           '--nproc-per-node=2', str(root / 'micro.py'), '--output', str(name.with_suffix('.json'))]
    with name.with_suffix('.log').open('x') as log:
        subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    ranks = [json.loads(name.with_name(name.name + f'-rank{rank}.json').read_text()) for rank in (0, 1)]
    expected = {(n, 7, i, r, t, layout) for n in range(1, 17) for i in (7, 9)
                for r in ('uniform', 'uneven', 'padded_zero') for t in ('i32', 'i64', 'mixed')
                for layout in ('contiguous', 'qkvz_split_6144')}
    required = ('eight_eager_and_graph_revisions_bits_exact', 'FULL_native_physical_state_bits_exact',
                'independent_native_graph_passed', 'whole_source_base_weights_metadata_readonly',
                'all_folded_replay_counter_cells_exact')
    for index, rank in enumerate(ranks):
        if not rank['completed'] or rank['rank'] != index or len(rank['records']) != 576:
            raise RuntimeError('Incomplete convolution component results')
        if {(r['n'], r['width'], r['index_stride'], r['route'], r['types'], r['layout'])
                for r in rank['records']} != expected:
            raise RuntimeError('Convolution component test matrix is incomplete')
        if not all(all(row[key] for key in required) for row in rank['records']):
            raise RuntimeError('Convolution component numerical validation failed')
        for filename, digest in rank['sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError('A convolution test input changed during validation')
    if any(hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest for filename, digest in hashes.items()):
        raise RuntimeError('A convolution source changed during validation')
    proof.update(passed=True, exact=True, ranks=ranks, sha256=hashes,
                 component_only_NOT_model_quality_or_end_to_end_speed_admission=True)
    with name.with_suffix('.primitive-audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-dir', required=True, type=Path)
    parser.add_argument('--backend-package', required=True, type=Path)
    args = parser.parse_args()
    validate(args.component_dir, args.backend_package)
    print('Convolution component proof passed; this is not an end-to-end result.')


if __name__ == '__main__':
    main()
