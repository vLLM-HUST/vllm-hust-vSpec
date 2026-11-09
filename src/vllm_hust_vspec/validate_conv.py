"""Validate experimental reference, stride and edge convolution components."""
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


def validate(component: str, root: Path, backend_package: Path, reference: Path | None = None) -> None:
    configs = {'reference': ('vspec_conv_tile', 'native-conv-strided-v3-r1'),
               'stride': ('vspec_conv_stride', 'native-stride-original-and-copy-controls-r1'),
               'edge': ('vspec_even_edge', 'even-edge-c46-r1')}
    if component not in configs:
        raise ValueError('Unsupported convolution component')
    lib, name = configs[component]
    root, backend_package = root.resolve(), backend_package.resolve()
    spec = importlib.util.find_spec('vllm_ascend')
    if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(backend_package):
        raise RuntimeError('Active vllm_ascend is not the explicitly selected new backend')
    sources = [root / item for item in ('micro.py', 'adapter.cpp', 'kernel.cpp', 'CMakeLists.txt',
               'native-kernel/causal_conv1d.h')]
    sources += sorted((root / 'native-kernel').rglob('*.h'))
    sources += [root / 'build-r1/lib' / f'lib{lib}_{suffix}.so' for suffix in ('adapter', 'kernel')]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError('Built convolution component or source missing')
    if component == 'stride':
        if reference is None:
            raise ValueError('Stride component requires a validated convolution reference directory')
        reference = reference.resolve()
        prior_path = reference / 'native-conv-strided-v3-r1-primitive-audit.json'
        prior = json.loads(prior_path.read_text())
        require_current_npu_proof(prior)
        if not prior['passed'] or not prior['exact'] or {r['rank'] for r in prior['ranks']} != {0, 1}:
            raise RuntimeError('Convolution reference proof did not pass')
        expected_prior = {(n, w, r, t, layout) for n in (1, 8, 14, 16) for w in (5, 7)
                          for r in ('uniform', 'uneven', 'padded_zero')
                          for t in ('i32', 'i64', 'mixed', 'wide_indices')
                          for layout in ('contiguous', 'qkvz_split_6144')}
        for rank in prior['ranks']:
            rows = rank['records']
            if not rank['completed'] or len(rows) != 192 or {
                    (r['rows'], r['width'], r['route'], r['types'], r['layout']) for r in rows} != expected_prior:
                raise RuntimeError('Convolution reference proof matrix is incomplete')
            if not all(r['all_eager_and_eight_graph_outputs_and_all_physical_state_bits_exact']
                       and r['independent_original_native_selfgraph_passed'] for r in rows):
                raise RuntimeError('Convolution reference numerical checks did not pass')
        for part in (prior, *prior['ranks']):
            for filename, digest in part['sha256'].items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                    raise RuntimeError('Convolution reference proof source changed')
        sources.append(prior_path)
        sources += [reference / 'build-r1/lib' / f'libvspec_conv_tile_{suffix}.so' for suffix in ('adapter', 'kernel')]
        if not all(path.is_file() for path in sources):
            raise FileNotFoundError('Validated reference libraries missing')
    runs = root / 'runs'
    if runs.exists():
        raise FileExistsError('Component output exists; use a fresh build directory')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    os.umask(0o077)
    runs.mkdir(mode=0o700)
    env = dict(os.environ)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = root / 'validation-cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    tmp = root / 'validation-tmp'
    tmp.mkdir(mode=0o700)
    env.update(TMPDIR=str(tmp), VLLM_PLUGINS='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', TORCH_DEVICE_BACKEND_AUTOLOAD='0')
    env.pop('PYTHONPATH', None)
    if reference is not None:
        env['VSPEC_CONV_REFERENCE_DIR'] = str(reference)
    output = runs / name
    cmd = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
           str(root / 'micro.py'), '--output', str(output.with_suffix('.json'))]
    with output.with_suffix('.log').open('x') as log:
        subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    ranks = [json.loads(output.with_name(output.name + f'-rank{rank}.json').read_text()) for rank in (0, 1)]
    routes, layouts = ('uniform', 'uneven', 'padded_zero'), ('contiguous', 'qkvz_split_6144')
    if component == 'edge':
        expected = {(n, w, r, t, layout) for n in range(1, 17) for w in (3, 9)
                    for r in routes for t in ('i32', 'i64', 'mixed') for layout in layouts}
        keys = ('n', 'width', 'route', 'types', 'layout')
        required = ('eight_eager_and_graph_revisions_bits_exact', 'FULL_native_physical_state_bits_exact',
                    'independent_native_graph_passed', 'whole_source_base_weights_metadata_readonly',
                    'all_folded_replay_counter_cells_exact')
    else:
        types = ('i32', 'i64', 'mixed', 'wide_indices')
        if component == 'stride':
            types += ('capacity8_indices',)
        expected = {(n, w, r, t, layout, state) for n in (1, 8, 14, 16) for w in (5, 7)
                    for r in routes for t in types for layout in layouts
                    for state in ((9, 11) if component == 'stride' else (9,))}
        keys = ('rows', 'width', 'route', 'types', 'layout')
        required = ('entire_qkvz_source_base_including_unused_z_NaN_and_signed_zero_immutable',
                    'all_eager_and_eight_graph_outputs_and_all_physical_state_bits_exact',
                    'independent_original_native_selfgraph_passed',
                    'unused_output_padding_and_unused_state_NaN_and_signed_zero_bits_checked',
                    'source_value_weight_and_metadata_unchanged',
                    'actual_replay_witness_folded_inside_same_kernel_all_cells_checked')
    for index, rank in enumerate(ranks):
        rows = rank['records']
        observed = {tuple(row[k] for k in keys) + ((row['state_len'] if component == 'stride' else 9,)
                    if component != 'edge' else ()) for row in rows}
        if not rank['completed'] or rank['rank'] != index or len(rows) != len(expected) or observed != expected:
            raise RuntimeError('Convolution component test matrix is incomplete')
        if not all(all(row[key] for key in required) for row in rows):
            raise RuntimeError('Convolution component numerical validation failed')
        if component == 'stride' and not all(row['candidate_uses_original_physical_input_stride_without_contiguous_copy']
                and row['old_copy2048_and_copy4096_controls_include_their_contiguous_copy'] for row in rows):
            raise RuntimeError('Convolution stride controls did not pass')
        for filename, digest in rank['sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError('A convolution test input changed')
    if any(hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest for filename, digest in hashes.items()):
        raise RuntimeError('A convolution source changed during validation')
    proof.update(passed=True, exact=True, ranks=ranks, sha256=hashes, cases_per_rank=len(expected),
                 component_only_no_model_or_quality_or_performance_admission=True)
    target = runs / (name + '.primitive-audit.json') if component == 'edge' else root / (name + '-primitive-audit.json')
    with target.open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('reference', 'stride', 'edge'), required=True)
    parser.add_argument('--component-dir', required=True, type=Path)
    parser.add_argument('--backend-package', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args()
    validate(args.component, args.component_dir, args.backend_package, args.reference_dir)
    print('Convolution component proof passed; this is not an end-to-end result.')


if __name__ == '__main__':
    main()
