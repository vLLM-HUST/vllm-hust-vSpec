import subprocess

import pytest

from vllm_hust_vspec.validate_mask import validate


def test_refuses_missing_backend_before_writing(tmp_path):
    destination = tmp_path / 'private'
    with pytest.raises(FileNotFoundError, match='reference kernel'):
        validate(tmp_path / 'missing', destination)
    assert not destination.exists()


def test_refuses_busy_devices_before_writing(tmp_path, monkeypatch):
    backend = tmp_path / 'backend'
    old = backend / 'vllm_ascend/ops/triton/q35_expert_mask.py'
    old.parent.mkdir(parents=True)
    old.write_text('# reference')
    destination = tmp_path / 'private'
    monkeypatch.setattr('vllm_hust_vspec.validate_mask.current_npu_allocation', lambda: [3, 7])
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: 'one NPU is busy')
    with pytest.raises(RuntimeError, match='idle'):
        validate(backend, destination)
    assert not destination.exists()
