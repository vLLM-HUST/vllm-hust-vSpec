from types import SimpleNamespace

import pytest

from vllm_hust_vspec.validate_prefix_clip import validate


def test_rejects_wrong_backend_before_creating_output(tmp_path, monkeypatch):
    destination = tmp_path / 'proof'
    monkeypatch.setattr('importlib.util.find_spec', lambda name: SimpleNamespace(origin='/other/backend/__init__.py'))
    with pytest.raises(RuntimeError, match='not the explicitly selected backend'):
        validate(tmp_path / 'backend', destination)
    assert not destination.exists()
