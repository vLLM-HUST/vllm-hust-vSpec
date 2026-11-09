from pathlib import Path
import subprocess

import pytest

from vllm_hust_vspec.backends.experimental_paths import artifact_dir, require_current_npu_proof


def test_artifact_directory_defaults_to_unconfigured(monkeypatch):
    monkeypatch.delenv('VSPEC_TEST_ARTIFACT_DIR', raising=False)
    path = artifact_dir('VSPEC_TEST_ARTIFACT_DIR')
    assert path == Path('/__vspec_artifacts_not_configured__/VSPEC_TEST_ARTIFACT_DIR')
    assert not path.exists()


def test_artifact_directory_uses_explicit_setting(monkeypatch, tmp_path):
    monkeypatch.setenv('VSPEC_TEST_ARTIFACT_DIR', str(tmp_path))
    assert artifact_dir('VSPEC_TEST_ARTIFACT_DIR') == tmp_path


def test_proof_must_match_detected_allocation(monkeypatch):
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('ASCEND_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw:
        ' 7 0 1 Ascend 910B2\n 3 0 0 Ascend 910B2\n 3 1 - Mcu\n')
    require_current_npu_proof({'physical_npus': [3, 7]})
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '0,1')
    require_current_npu_proof({'physical_npus': [3, 7]})
    with pytest.raises(RuntimeError, match='does not match'):
        require_current_npu_proof({'physical_npus': [3, 8]})


def test_proof_rejects_ambiguous_allocation(monkeypatch):
    monkeypatch.delenv('ASCEND_VISIBLE_DEVICES', raising=False)
    monkeypatch.setenv('ASCEND_RT_VISIBLE_DEVICES', '1,0')
    with pytest.raises(RuntimeError, match='prevents validating'):
        require_current_npu_proof({'physical_npus': [3, 7]})
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES')
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: ' 3 0 0 Ascend 910B2\n')
    with pytest.raises(RuntimeError, match='exactly two'):
        require_current_npu_proof({'physical_npus': [3, 7]})
