from pathlib import Path

import pytest

from vllm_hust_vspec import validate_two_bank


def test_requires_checkpoint_before_allocating_output(tmp_path):
    destination = tmp_path / 'proof'
    with pytest.raises(FileNotFoundError):
        validate_two_bank.validate(tmp_path / 'missing-model', destination)
    assert not destination.exists()


def test_rejects_existing_or_model_nested_destination(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors.index.json').write_text('{}')
    existing = tmp_path / 'existing'
    existing.mkdir()
    for destination in (existing, model / 'proof'):
        with pytest.raises(ValueError):
            validate_two_bank.validate(model, destination)
    assert not (model / 'proof').exists()
