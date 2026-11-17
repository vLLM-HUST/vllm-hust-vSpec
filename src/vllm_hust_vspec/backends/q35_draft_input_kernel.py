"""No-communication proposal-input primitives; original weights remain intact."""

import torch
import torch.nn.functional as F


def embedding(input_ids, full_weight):
    # Native TP2 sums one owner plus BF16 positive zero. Keep zero sign
    # canonicalization rather than assuming direct lookup is bit-identical.
    return F.embedding(input_ids.long(), full_weight) + 0.


def split_fc(hidden, weights):
    # Preserve both original projection matrix shapes and K ordering.
    return torch.cat([F.linear(hidden, weight) for weight in weights], dim=-1)


def full_fc(hidden, weight):
    return F.linear(hidden, weight)
