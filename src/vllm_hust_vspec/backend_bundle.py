"""Prepare the pinned experimental Ascend backend without altering its source checkout.

This is a staging operation only; it does not install, enable, or validate a
backend. The original runtime safety gates remain in force.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess

BASE_COMMIT = '4e57439e58ed3d78e675f9fd7b4614fb183c5394'
BUNDLE = Path(__file__).resolve().parent / '_vendor' / 'ascend_hust'


def prepare(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'destination already exists: {destination}')
    destination = destination.resolve(strict=False)
    if destination.is_relative_to(source):
        raise ValueError('destination must not be inside the source checkout')
    subprocess.run(['git', '-C', str(source), 'cat-file', '-e', f'{BASE_COMMIT}^{{commit}}'], check=True)
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout', str(source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', BASE_COMMIT], check=True)
    patch = BUNDLE / 'tracked-changes.patch'
    subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(patch)], check=True)
    subprocess.run(['git', '-C', str(destination), 'apply', str(patch)], check=True)
    overlay = BUNDLE / 'overlay'
    for origin in sorted(overlay.rglob('*.py')):
        if origin.is_symlink():
            raise ValueError(f'overlay symlink is not allowed: {origin}')
        target = destination / origin.relative_to(overlay)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'overlay would overwrite an existing file: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='local Git checkout containing the pinned base')
    parser.add_argument('--destination', type=Path, required=True, help='new, private output directory')
    args = parser.parse_args()
    prepare(args.source, args.destination)
    print(f'Prepared experimental backend source at {args.destination}; not installed or device-validated.')


if __name__ == '__main__':
    main()
