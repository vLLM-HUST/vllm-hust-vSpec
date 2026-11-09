"""Prepare the pinned experimental Ascend backend without altering its source checkout.

This is a staging operation only; it does not install, enable, or validate a
backend. The original runtime safety gates remain in force.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

BASE_COMMIT = '4e57439e58ed3d78e675f9fd7b4614fb183c5394'
BUNDLE = Path(__file__).resolve().parent / '_vendor' / 'ascend_hust'
THIRD_PARTY_ARCHIVES = {
    'abseil-cpp-20230802.1.tar.gz': '987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed',
    'protobuf-25.1.tar.gz': '9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42',
    'include.zip': 'a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d',
}


def verified_archives(source: Path) -> list[tuple[Path, str]]:
    """Locate offline CANN dependencies absent from the pinned Git commit."""
    package_dir = source / 'csrc' / 'third_party' / 'pkg'
    archives = []
    for name, digest in THIRD_PARTY_ARCHIVES.items():
        archive = package_dir / name
        if archive.is_symlink() or not archive.is_file():
            raise FileNotFoundError(f'missing offline backend dependency: {archive}')
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise ValueError(f'offline backend dependency hash mismatch: {archive}')
        archives.append((archive, name))
    return archives


def prepare(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'destination already exists: {destination}')
    destination = destination.resolve(strict=False)
    if destination.is_relative_to(source):
        raise ValueError('destination must not be inside the source checkout')
    archives = verified_archives(source)
    subprocess.run(['git', '-C', str(source), 'cat-file', '-e', f'{BASE_COMMIT}^{{commit}}'], check=True)
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', '--no-checkout', str(source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', BASE_COMMIT], check=True)
    for name in ('tracked-changes.patch', 'build-python.patch'):
        patch = BUNDLE / name
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
    package_dir = destination / 'csrc' / 'third_party' / 'pkg'
    package_dir.mkdir(parents=True, exist_ok=True)
    for origin, name in archives:
        target = package_dir / name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'offline dependency would overwrite an existing file: {target}')
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
