"""Identical native BF16 linear; only immutable weight layout may differ."""
from torch.nn.functional import linear


def execute(value, weight): return linear(value, weight)
