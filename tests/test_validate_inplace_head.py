import pytest

from vllm_hust_vspec import validate_inplace_head


def test_requires_checkpoint_and_preserves_destination(tmp_path):
    destination = tmp_path / 'proof'
    with pytest.raises(FileNotFoundError):
        validate_inplace_head.validate(tmp_path / 'missing', destination)
    assert not destination.exists()


def test_rejects_output_inside_checkpoint(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors.index.json').write_text('{}')
    with pytest.raises(ValueError):
        validate_inplace_head.validate(model, model / 'proof')
