"""Stage and build an experimental Ascend operator in a new directory.

Building is not validation, registration or activation of the operator.
"""
from __future__ import annotations

import argparse
import importlib.resources
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def build_operator(source: Path, destination: Path, cann_root: Path,
                   python: Path, torch_dir: Path, jobs: int = 2) -> None:
    source, destination = source.resolve(), destination.resolve()
    cann_root, python, torch_dir = cann_root.resolve(), python.absolute(), torch_dir.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Destination must be a new directory outside the packaged source')
    if not (source / 'CMakeLists.txt').is_file():
        raise FileNotFoundError('Packaged operator source missing')
    if not (cann_root / 'aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake').is_file():
        raise FileNotFoundError('An installed aarch64 CANN SDK with AscendC CMake support is required')
    if not (torch_dir / 'TorchConfig.cmake').is_file():
        raise FileNotFoundError('Torch CMake configuration not found')
    if jobs < 1:
        raise ValueError('jobs must be positive')
    shutil.copytree(source, destination)
    destination.chmod(0o700)
    build = destination / 'build-r1'
    env = dict(os.environ, ASCEND_HOME_PATH=str(cann_root))
    subprocess.run(['cmake', '-S', str(destination), '-B', str(build),
                    '-DCMAKE_BUILD_TYPE=Release', f'-DTorch_DIR={torch_dir}',
                    f'-DPython3_EXECUTABLE={python}'], env=env, check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', str(jobs)], env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path,
                        help='New private directory for source and build output')
    parser.add_argument('--cann-root', required=True, type=Path, help='Installed CANN SDK root')
    parser.add_argument('--operator', choices=('gamma6-tail', 'gamma6-rgdr', 'gamma6-rgdr-reference',
                                               'gamma6-rgdr-packed-reference', 'gamma8-rgdr',
                                               'conv-reference', 'conv-stride', 'conv-edge'),
                        default='gamma6-tail')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec is None or torch_spec.origin is None:
        raise RuntimeError('Install a compatible PyTorch before building')
    torch_dir = Path(torch_spec.origin).parent / 'share/cmake/Torch'
    component = args.operator.replace('-', '_')
    source = Path(str(importlib.resources.files('vllm_hust_vspec') / f'_vendor/native/{component}'))
    build_operator(source, args.destination, args.cann_root, Path(sys.executable), torch_dir, args.jobs)
    print('Operator built. It has not been correctness-validated or enabled.')


if __name__ == '__main__':
    main()
