"""Validate the experimental GDN-input fusion on two Ascend devices."""
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

from .backends.experimental_paths import current_npu_allocation, require_current_npu_proof


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(width: int, backend: Path, vllm: Path, destination: Path) -> None:
    backend, vllm, destination = backend.resolve(), vllm.resolve(), destination.resolve()
    if width not in (6, 8):
        raise ValueError('Unsupported experimental width')
    if destination.exists() or destination.is_relative_to(backend) or destination.is_relative_to(vllm):
        raise ValueError('Destination must be new and outside both source trees')
    for module, tree in (('vllm_ascend', backend), ('vllm', vllm)):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(tree):
            raise RuntimeError(f'Active {module} is not the explicitly selected source')
    ascend = backend / 'vllm_ascend'
    files = [ascend / p for p in ('ops/gdn.py', 'ops/gdn_input_fusion.py',
             'ops/triton/q35_fused_gdn_inputs.py', 'ops/triton/fused_gdn_gating.py', 'device/device_op.py')]
    files += [vllm / p for p in ('vllm/third_party/flash_linear_attention/ops/l2norm.py',
              'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py')]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError('Required backend or vLLM reference sources are missing')
    physical = current_npu_allocation()
    if subprocess.check_output(['npu-smi', 'info'], text=True, timeout=15).count('No running processes found') != 2:
        raise RuntimeError('The two allocated NPUs must be idle before this component test')
    package = Path(str(importlib.resources.files('vllm_hust_vspec') /
                       f'_vendor/operators/gamma{width}_input'))
    if sha(package / 'kernel.py') != sha(files[2]):
        raise RuntimeError('Packaged fusion kernel differs from the selected backend')
    destination.mkdir(parents=True, mode=0o700)
    for name in ('kernel.py', 'micro_fusion.py', 'reference.py'):
        shutil.copy2(package / name, destination / name)
    paths = [destination / name for name in ('kernel.py', 'micro_fusion.py', 'reference.py')] + files
    hashes = {str(p): sha(p) for p in paths}
    with (destination / 'launch.json').open('x') as stream:
        json.dump({'sha256': hashes, 'component_only': True, 'physical_npus': physical}, stream, indent=2)
    env = dict(os.environ)
    for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'VLLM_CACHE_ROOT', 'TRITON_CACHE_DIR',
                'TORCHINDUCTOR_CACHE_DIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'HF_HOME',
                'ASCEND_WORK_PATH', 'ASCEND_PROCESS_LOG_PATH', 'ASCEND_CACHE_PATH', 'ACL_OP_COMPILER_CACHE_DIR'):
        path = destination / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env[key] = str(path)
    tmp = destination / 'tmp'
    tmp.mkdir(mode=0o700)
    env.update(TMPDIR=str(tmp), PYTHONPATH=str(destination), VLLM_PLUGINS='',
               VSPEC_VLLM_SOURCE_ROOT=str(vllm), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    ranks = []
    for rank in (0, 1):
        output = destination / f'rank{rank}.json'
        command = [sys.executable, '-B', str(destination / 'micro_fusion.py'), '--device', str(rank),
                   '--output', str(output)]
        with (destination / f'rank{rank}.log').open('x') as log:
            subprocess.run(command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(output.read_text())
        rows = [case for case in data['cases'] if 'phase' in case]
        required = ('original_native_eager_self_graph_bits_exact',
                    'actual_consumed_BF16_bits_including_signed_zero_exact', 'canaries_unchanged',
                    'inputs_unchanged', 'fused_eager_self_graph_bits_exact',
                    'complete_all_five_inputs_raw_bits_readonly')
        if not data['passed'] or data['device'] != rank or len(rows) != 128:
            raise RuntimeError('Incomplete GDN input component results')
        if {row['tokens'] for row in rows} != {n * (width + 1) for n in range(1, 17)}:
            raise RuntimeError('GDN input component token matrix is incomplete')
        if not all(all(row[key] for key in required) and row['fp32_gate_rtol'] == 1e-6
                   and row['fp32_gate_atol'] == 1e-6 and row['replays'] == 3 for row in rows):
            raise RuntimeError('GDN input component numerical validation failed')
        ranks.append(data)
    proof = {'physical_npus': physical}
    require_current_npu_proof(proof)
    if any(sha(Path(name)) != digest for name, digest in hashes.items()):
        raise RuntimeError('A GDN input test source changed during validation')
    proof.update(completed=True, component_only=True, eligible_tokens=[n * (width + 1) for n in range(1, 17)],
                 ranks=ranks, sha256=hashes, no_model_speed_quality_or_goal_admission=True,
                 **{f'only_gamma{width}_all16_batches': True})
    with (destination / f'gamma{width}-input-fusion-c46-primitive-audit-r1.json').open('x') as stream:
        json.dump(proof, stream, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, choices=(6, 8), required=True)
    parser.add_argument('--backend', required=True, type=Path)
    parser.add_argument('--vllm-source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    validate(args.width, args.backend, args.vllm_source, args.destination)
    print('GDN input component proof passed; this is not an end-to-end result.')


if __name__ == '__main__':
    main()
