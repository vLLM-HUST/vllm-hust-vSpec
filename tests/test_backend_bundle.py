from pathlib import Path
import hashlib
import subprocess

import pytest

from vllm_hust_vspec import backend_bundle


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def test_prepares_copy_without_changing_source(monkeypatch, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-q')
    tracked = source / 'tracked.txt'
    tracked.write_text('before\n')
    build_file = source / 'setup.py'
    build_file.write_text('original\n')
    git(source, 'add', 'tracked.txt', 'setup.py')
    git(source, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'base')
    commit = git(source, 'rev-parse', 'HEAD')
    tracked.write_text('after\n')
    bundle = tmp_path / 'bundle'
    overlay = bundle / 'overlay' / 'vllm_ascend'
    overlay.mkdir(parents=True)
    (overlay / 'extra.py').write_text('value = 1\n')
    (bundle / 'tracked-changes.patch').write_text(git(source, 'diff', '--', 'tracked.txt') + '\n')
    build_file.write_text('using-build-python\n')
    (bundle / 'build-python.patch').write_text(git(source, 'diff', '--', 'setup.py') + '\n')
    archive_dir = source / 'csrc' / 'third_party' / 'pkg'
    archive_dir.mkdir(parents=True)
    archives = {'offline.tar.gz': b'pinned offline source'}
    for name, payload in archives.items():
        (archive_dir / name).write_bytes(payload)
    monkeypatch.setattr(backend_bundle, 'BASE_COMMIT', commit)
    monkeypatch.setattr(backend_bundle, 'BUNDLE', bundle)
    monkeypatch.setattr(backend_bundle, 'THIRD_PARTY_ARCHIVES', {
        name: hashlib.sha256(payload).hexdigest() for name, payload in archives.items()
    })
    output = tmp_path / 'prepared'
    backend_bundle.prepare(source, output)
    assert (output / 'tracked.txt').read_text() == 'after\n'
    assert (output / 'setup.py').read_text() == 'using-build-python\n'
    assert (output / 'vllm_ascend' / 'extra.py').read_text() == 'value = 1\n'
    assert (output / 'csrc' / 'third_party' / 'pkg' / 'offline.tar.gz').read_bytes() == archives['offline.tar.gz']
    assert tracked.read_text() == 'after\n'
    assert git(source, 'rev-parse', 'HEAD') == commit
    with pytest.raises(FileExistsError):
        backend_bundle.prepare(source, output)
    with pytest.raises(ValueError):
        backend_bundle.prepare(source, source / 'must-not-write-here')
    (archive_dir / 'offline.tar.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        backend_bundle.prepare(source, tmp_path / 'bad')
