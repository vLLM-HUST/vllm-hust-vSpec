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

from .backends.experimental_paths import configure_custom_opp, current_npu_allocation, require_current_npu_proof


def _read_proof(path: Path) -> dict:
    proof = json.loads(path.read_text())
    require_current_npu_proof(proof)
    if not proof.get('passed') or not proof.get('exact'):
        raise RuntimeError('Recurrent reference proof did not pass')
    ranks = proof.get('ranks', [])
    expected = {(n, bank, seed) for n in range(1, 17) for bank in (192, 647) for seed in range(6)}
    if {rank.get('rank') for rank in ranks} != {0, 1}:
        raise RuntimeError('Recurrent reference proof does not cover both ranks')
    for rank in ranks:
        rows = rank.get('records', [])
        if not rank.get('completed') or len(rows) != 192 or {(r['requests'], r['bank'], r['seed']) for r in rows} != expected:
            raise RuntimeError('Recurrent reference proof matrix is incomplete')
        if not all(r['independent_native_eager_graph_reference'] and r['eight_eager_graph_revisions_bits_exact']
                   and r['full_FP32_state_including_untouched_NaN_payload_and_signed_zero_exact'] for r in rows):
            raise RuntimeError('Recurrent reference numerical checks did not pass')
    for part in (proof, *proof['ranks']):
        for name, digest in part['sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Recurrent reference proof source changed')
    return proof


def _check_packed_source(root: Path, reference: Path, tagged: bool) -> None:
    source = (root / 'rgdr_full128.h').read_text()
    packed_broadcast = '''        // Exact bit replication into ONE32B block per delta, not128 columns.
        // Each FMA repeat reads that block with src0 block stride0.
        Brcb(broadTmpInUb, deltaInUb, static_cast<uint8_t>(Ceil(curSingleV, 8)), {1, 8});'''
    original_broadcast = '        Broadcast<float, 2, 1>(broadTmpInUb, deltaInUb, stateShape, deltaShape); //  2: Dim Number 1: Second Dim'
    packed_fma = '''        for (uint32_t column = 0; column < alignK_; column += REPEAT_LENTH) {
            const uint64_t mask = Std::min(REPEAT_LENTH, alignK_ - column);
            MulAddDst(stateInUb[column], broadTmpInUb, kInUb[curQKOffset + column], mask,
                static_cast<uint8_t>(curSingleV), {1, 0, 1, static_cast<uint8_t>(alignK_ / FP32_NUM_PER_BLOCK), 1, 0});
        }'''
    if source.count(packed_broadcast) != 1 or source.count(packed_fma) != 1:
        raise RuntimeError('Packed arithmetic differs from the admitted transformations')
    source = source.replace(packed_broadcast, original_broadcast).replace(
        packed_fma, '        MatVecMul(broadTmpInUb, kInUb[curQKOffset], stateInUb, curSingleV, true);')
    if source != (reference / 'rgdr_full128.h').read_text():
        raise RuntimeError('Packed arithmetic differs from the reference source')
    if (root / 'recurrent_gated_delta_rule_tiling_data.h').read_bytes() != (reference / 'recurrent_gated_delta_rule_tiling_data.h').read_bytes():
        raise RuntimeError('Packed tiling differs from the reference')
    name = 'vspec_rgdr_pack6_tag' if tagged else 'vspec_rgdr_pack6'
    kernel = (root / 'kernel.cpp').read_text().replace(name, 'vspec_rgdr_tail6')
    if tagged:
        kernel = kernel.replace('counts[17 + tokens / 7]', 'counts[tokens / 7]')
    if kernel != (reference / 'kernel.cpp').read_text():
        raise RuntimeError('Packed kernel changes more than the admitted counter destination')
    if (root / 'CMakeLists.txt').read_text().replace(name, 'vspec_rgdr_tail6') != (reference / 'CMakeLists.txt').read_text():
        raise RuntimeError('Packed build configuration differs from the reference')
    adapter = (root / 'adapter.cpp').read_text().replace(name, 'vspec_rgdr_tail6')
    adapter = adapter.replace('    TORCH_CHECK(v_step == 128 && output_buffers == 1, "Original128 single-bank path with exact packed-delta FMA only");',
        '    TORCH_CHECK((v_step == 128 && output_buffers == 1)\n                || (v_step == 64 && output_buffers == 2), "Two UB-bounded pinned pipeline variants only");')
    if tagged:
        adapter = adapter.replace('IntArrayRef({34})', 'IntArrayRef({17})').replace(
            'Single folded add in independent packed-kernel diagnostic cells17+N only', 'Folded replay counters only')
    if adapter != (reference / 'adapter.cpp').read_text():
        raise RuntimeError('Packed adapter changes more than the admitted guards')


def validate(component: str, root: Path, backend_package: Path,
             reference: Path | None = None, prior: Path | None = None) -> None:
    configs = {
        'gamma6-reference': ('vspec_rgdr_tail6', 'gamma6-tail-c46-r1', 7),
        'gamma6-packed-reference': ('vspec_rgdr_pack6', 'gamma6-packed-delta-c46-r1', 7),
        'gamma6-packed': ('vspec_rgdr_pack6_tag', 'gamma6-tail-c46-r1', 7),
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
    packed = component.startswith('gamma6-packed')
    if packed:
        if reference is None:
            raise ValueError('Packed recurrent operators require a validated gamma-six reference directory')
        reference = reference.resolve()
        ref_proof = reference / 'runs/gamma6-tail-c46-r1.primitive-audit.json'
        _read_proof(ref_proof)
        sources += [ref_proof, *(reference / name for name in ('rgdr_full128.h',
                    'recurrent_gated_delta_rule_tiling_data.h', 'kernel.cpp', 'adapter.cpp', 'CMakeLists.txt'))]
        _check_packed_source(root, reference, component == 'gamma6-packed')
        if component == 'gamma6-packed':
            if prior is None:
                raise ValueError('Tagged recurrent operator requires a validated packed-reference directory')
            prior = prior.resolve()
            prior_proof = prior / 'runs/gamma6-packed-delta-c46-r1.primitive-audit.json'
            _read_proof(prior_proof)
            if (prior / 'rgdr_full128.h').read_bytes() != (root / 'rgdr_full128.h').read_bytes():
                raise RuntimeError('Tagged packed header differs from its validated predecessor')
            sources += [prior_proof, prior / 'rgdr_full128.h']
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
    configure_custom_opp(env, backend_package)
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
    if packed:
        env['VSPEC_GAMMA6_RGDR_REFERENCE_DIR'] = str(reference)
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
        if packed and not all(row['twelve_continuous_state_replays_WITHOUT_state_reset_full_bits_exact'] for row in rank['records']):
            raise RuntimeError('Packed recurrent continuous-state validation failed')
        if component == 'gamma6-packed' and not all(row['one_existing_atomic_add_targets_independent_cells17_plus_N_to_detect_wrong_old_kernel'] for row in rank['records']):
            raise RuntimeError('Tagged recurrent counter validation failed')
        for filename, digest in rank['sha256'].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise RuntimeError('A recurrent test input changed during validation')
    if any(hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest for filename, digest in hashes.items()):
        raise RuntimeError('A recurrent source changed during validation')
    proof.update(passed=True, exact=True, ranks=ranks, sha256=hashes,
                 component_only_NOT_model_quality_or_end_to_end_speed_admission=True)
    if component == 'gamma6-packed':
        proof.update(exact_same_packed_arithmetic_header_as_proven_v1=True,
                     ONE_existing_folded_diagnostic_add_to_cells17_plus_N_no_extra_hot_operation=True)
    with name.with_suffix('.primitive-audit.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('gamma6-reference', 'gamma6-packed-reference', 'gamma6-packed', 'gamma8'), required=True)
    parser.add_argument('--component-dir', required=True, type=Path)
    parser.add_argument('--backend-package', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--prior-dir', type=Path)
    args = parser.parse_args()
    validate(args.component, args.component_dir, args.backend_package, args.reference_dir, args.prior_dir)
    print('Recurrent component proof passed; this is not an end-to-end result.')


if __name__ == '__main__':
    main()
