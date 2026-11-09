from pathlib import Path
import subprocess

import pytest

from vllm_hust_vspec.native_operator_build import build_operator


def test_stages_source_and_invokes_build_without_overwriting(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'CMakeLists.txt').write_text('project(test)')
    cann = tmp_path / 'cann'
    cmake = cann / 'aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake'
    cmake.parent.mkdir(parents=True)
    cmake.write_text('')
    torch = tmp_path / 'torch'
    torch.mkdir()
    (torch / 'TorchConfig.cmake').write_text('')
    python = tmp_path / 'env/bin/python'
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    destination = tmp_path / 'private-build'
    build_operator(source, destination, cann, python, torch, 3)
    assert (destination / 'CMakeLists.txt').read_text() == 'project(test)'
    assert len(calls) == 2 and calls[0][1]['env']['ASCEND_HOME_PATH'] == str(cann)
    assert f'-DPython3_EXECUTABLE={python}' in calls[0][0]
    assert calls[1][0][-2:] == ['--parallel', '3']
    with pytest.raises(ValueError, match='new directory'):
        build_operator(source, destination, cann, python, torch)


def test_missing_sdk_fails_before_creating_destination(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'CMakeLists.txt').write_text('project(test)')
    destination = tmp_path / 'private-build'
    with pytest.raises(FileNotFoundError, match='CANN SDK'):
        build_operator(source, destination, tmp_path / 'missing', Path('/usr/bin/python3'), tmp_path / 'torch')
    assert not destination.exists()
