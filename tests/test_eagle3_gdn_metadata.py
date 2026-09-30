from __future__ import annotations

from types import SimpleNamespace

from vllm_hust_vspec.backends.eagle3_gdn_metadata import (
    _is_uniform_spec_decode,
)


def test_uniform_spec_decode_requires_every_request_on_spec_path() -> None:
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=16,
        spec_sequence_masks=object(),
        spec_query_start_loc=object(),
        spec_state_indices_tensor=object(),
        num_accepted_tokens=object(),
    )

    assert _is_uniform_spec_decode(metadata, 16)
    metadata.num_spec_decodes = 15
    assert not _is_uniform_spec_decode(metadata, 16)
    metadata.num_spec_decodes = 16
    metadata.num_prefills = 1
    assert not _is_uniform_spec_decode(metadata, 16)
