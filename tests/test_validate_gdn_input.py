from types import SimpleNamespace

import pytest

from vllm_hust_vspec.validate_gdn_input import validate


def test_rejects_invalid_width_before_writing(tmp_path):
    destination = tmp_path / 'proof'
    with pytest.raises(ValueError, match='width'):
        validate(4, tmp_path / 'backend', tmp_path / 'vllm', destination)
    assert not destination.exists()


def test_rejects_unselected_backend_before_writing(tmp_path, monkeypatch):
    destination = tmp_path / 'proof'
    monkeypatch.setattr('importlib.util.find_spec', lambda name: SimpleNamespace(origin='/other/package/__init__.py'))
    with pytest.raises(RuntimeError, match='not the explicitly selected source'):
        validate(6, tmp_path / 'backend', tmp_path / 'vllm', destination)
    assert not destination.exists()
