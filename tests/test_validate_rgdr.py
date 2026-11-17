from types import SimpleNamespace
from pathlib import Path
import shutil

import pytest

from vllm_hust_vspec.validate_rgdr import _check_packed_source, validate


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


def test_packed_source_contract_detects_unexpected_changes(tmp_path):
    vendor = Path(__file__).resolve().parents[1] / 'src/vllm_hust_vspec/_vendor/native'
    reference = vendor / 'gamma6_rgdr_reference'
    _check_packed_source(vendor / 'gamma6_rgdr_packed_reference', reference, False)
    _check_packed_source(vendor / 'gamma6_rgdr', reference, True)
    changed = tmp_path / 'changed'
    shutil.copytree(vendor / 'gamma6_rgdr', changed)
    with (changed / 'kernel.cpp').open('a') as stream:
        stream.write('\n// unexpected change\n')
    with pytest.raises(RuntimeError, match='kernel changes'):
        _check_packed_source(changed, reference, True)
