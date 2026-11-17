"""Diagnostic attention comparison restricted to real request query rows."""

import torch


def attention_comparison(candidate, reference, actual_tokens):
    """Never ignore a real-row NaN or mismatch; dummy graph rows are unused."""
    live = (torch.arange(candidate.shape[0], device=candidate.device) < actual_tokens).view(-1, 1, 1)
    candidate_finite, reference_finite = torch.isfinite(candidate), torch.isfinite(reference)
    equal = candidate_finite & reference_finite & torch.isclose(candidate, reference, rtol=0.025, atol=0.003)
    passed = torch.where(live, equal, True).all()
    error = torch.where(live, (candidate.float() - reference.float()).abs(), 0).amax()
    statistics = torch.stack((
        error,
        (live & ~candidate_finite).sum(dtype=torch.int32).float(),
        (live & ~reference_finite).sum(dtype=torch.int32).float(),
        (~live & ~candidate_finite).sum(dtype=torch.int32).float(),
        (~live & ~reference_finite).sum(dtype=torch.int32).float(),
    ))
    return passed, statistics
