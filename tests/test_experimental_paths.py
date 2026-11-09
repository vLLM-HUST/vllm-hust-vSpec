from pathlib import Path

from vllm_hust_vspec.backends.experimental_paths import artifact_dir


def test_artifact_directory_defaults_to_unconfigured(monkeypatch):
    monkeypatch.delenv('VSPEC_TEST_ARTIFACT_DIR', raising=False)
    path = artifact_dir('VSPEC_TEST_ARTIFACT_DIR')
    assert path == Path('/__vspec_artifacts_not_configured__/VSPEC_TEST_ARTIFACT_DIR')
    assert not path.exists()


def test_artifact_directory_uses_explicit_setting(monkeypatch, tmp_path):
    monkeypatch.setenv('VSPEC_TEST_ARTIFACT_DIR', str(tmp_path))
    assert artifact_dir('VSPEC_TEST_ARTIFACT_DIR') == tmp_path
