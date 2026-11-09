from types import SimpleNamespace

import pytest

from vllm_hust_vspec.validate_tail_conv import validate


def test_rejects_backend_mismatch_before_creating_output(tmp_path, monkeypatch):
    root = tmp_path / 'component'
    root.mkdir()
    monkeypatch.setattr('importlib.util.find_spec', lambda name: SimpleNamespace(origin='/other/vllm_ascend/__init__.py'))
    with pytest.raises(RuntimeError, match='not the explicitly selected new backend'):
        validate(root, tmp_path / 'chosen-backend')
    assert not (root / 'runs').exists()
