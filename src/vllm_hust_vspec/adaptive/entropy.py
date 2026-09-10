"""Token-level confidence stopping for serial Draft proposals."""

from __future__ import annotations

import math
from typing import Any


def topk_entropy(logits: Any, topk: int) -> Any:
    """Return row-wise entropy in nats over the largest logits."""
    import torch

    if not torch.is_tensor(logits) or logits.ndim < 2:
        raise ValueError("logits must be a tensor with at least two dimensions")
    candidates = min(topk, int(logits.shape[-1]))
    if candidates < 2:
        return torch.zeros(logits.shape[:-1], device=logits.device)
    values = torch.topk(logits.float(), k=candidates, dim=-1).values
    log_probabilities = torch.log_softmax(values, dim=-1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1)


def normalized_topk_entropy(logits: Any, topk: int) -> Any:
    """Return row-wise top-k entropy normalized to [0, 1]."""
    candidates = min(topk, int(logits.shape[-1]))
    if candidates < 2:
        return topk_entropy(logits, topk)
    return (topk_entropy(logits, topk) / math.log(candidates)).clamp_(0.0, 1.0)


class EntropyDraftStopper:
    """Choose a current-round stop position independently for each request."""

    def __init__(
        self,
        *,
        max_gamma: int,
        min_gamma: int,
        threshold: float = 0.3,
        scale: float = 0.15,
    ) -> None:
        if not 0 <= min_gamma <= max_gamma:
            raise ValueError("min_gamma must be in [0, max_gamma]")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self.threshold = threshold
        self.scale = scale

    def stop_lengths(self, entropies: Any) -> Any:
        """Return one effective Draft length per request without a host sync.

        ``entropies`` is shaped ``[draft_position, batch]``. A request keeps the
        token at the first uncertain position and stops before generating its
        successor, so every non-empty proposal contains at least one token.
        """
        import torch

        if not torch.is_tensor(entropies) or entropies.ndim != 2:
            raise ValueError("entropies must have shape [position, batch]")
        gamma, batch_size = entropies.shape
        if gamma <= 0 or gamma > self.max_gamma:
            raise ValueError("entropy position count is outside configured gamma")
        uncertain = entropies * self.scale > self.threshold * self.threshold
        has_stop = uncertain.any(dim=0)
        first_stop = uncertain.to(torch.int32).argmax(dim=0) + 1
        full_length = torch.full(
            (batch_size,),
            gamma,
            dtype=first_stop.dtype,
            device=first_stop.device,
        )
        return torch.where(has_stop, first_stop, full_length)

    def mask_draft_tokens(self, draft_token_ids: Any, entropies: Any) -> Any:
        """Replace each request's current-round Draft suffix with placeholders."""
        import torch

        if not torch.is_tensor(draft_token_ids) or draft_token_ids.ndim != 2:
            raise ValueError("draft_token_ids must have shape [batch, gamma]")
        batch_size, gamma = draft_token_ids.shape
        if entropies.shape != (gamma, batch_size):
            raise ValueError(
                "entropy shape must match transposed Draft token shape: "
                f"entropy={tuple(entropies.shape)}, draft={tuple(draft_token_ids.shape)}"
            )
        lengths = self.stop_lengths(entropies)
        positions = torch.arange(gamma, device=draft_token_ids.device)
        invalid = positions.unsqueeze(0) >= lengths.unsqueeze(1)
        return draft_token_ids.masked_fill(invalid, -1), lengths
