"""Opaque validation side effect: actual row metadata is read outside Dynamo.

Only validation counters/error flags mutate. No input/output/model tensor is
modified, no host read of tensor values occurs, and ON never calls this op.
"""

import torch


@torch.library.custom_op('hust_vspec::draft_input_shadow', mutates_args=('counts', 'errors'))
def record_validation(actual: torch.Tensor, expected: torch.Tensor,
                      counts: torch.Tensor, errors: torch.Tensor, feature: int) -> None:
    rows = actual.shape[0]
    flag = ((actual.view(torch.int16) != expected.view(torch.int16))
            | ~torch.isfinite(actual) | ~torch.isfinite(expected)).any().to(torch.int32)
    counts[rows, feature].add_(1)
    torch.maximum(errors[rows, feature], flag, out=errors[rows, feature])


@record_validation.register_fake
def fake_record_validation(actual, expected, counts, errors, feature):
    return None
