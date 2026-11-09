from types import SimpleNamespace

import pytest

from vllm_hust_vspec.validate_rgdr import validate


def test_rejects_unknown_operator_before_writing(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        validate('unknown', tmp_path / 'component', tmp_path / 'backend')


def test_rejects_unselected_backend_before_writing(tmp_path, monkeypatch):
    root = tmp_path / 'component'
    root.mkdir()
    monkeypatch.setattr('importlib.util.find_spec', lambda name: SimpleNamespace(origin='/other/backend/__init__.py'))
    with pytest.raises(RuntimeError, match='not the explicitly selected new backend'):
        validate('gamma8', root, tmp_path / 'backend')
    assert not (root / 'runs').exists()
