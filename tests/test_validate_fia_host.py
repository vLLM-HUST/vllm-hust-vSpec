import pytest

from vllm_hust_vspec import validate_fia_host


def test_requires_built_operator_before_proof(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_fia_host.validate(tmp_path)
    assert not (tmp_path / 'primitive-native-audit-r1.json').exists()


def test_does_not_overwrite_existing_proof(tmp_path):
    proof = tmp_path / 'primitive-native-audit-r1.json'
    proof.write_text('original')
    with pytest.raises(ValueError):
        validate_fia_host.validate(tmp_path)
    assert proof.read_text() == 'original'
