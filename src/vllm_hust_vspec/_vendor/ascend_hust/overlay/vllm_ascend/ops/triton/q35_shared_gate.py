"""Retain native BF16 sigmoid rounding before the BF16 gate multiply."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid(INPUT, OUTPUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(INPUT + index, mask=index < SIZE, other=0).to(tl.float32)
    tl.store(OUTPUT + index, tl.sigmoid(value), mask=index < SIZE)


@triton.jit
def _gate_multiply(GATE, VALUE, OUTPUT, TOKENS: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, 2048)
    valid = row < TOKENS
    gate = tl.load(GATE + row, mask=valid, other=0).to(tl.float32)
    offset = row[:, None] * 2048 + column[None, :]
    value = tl.load(VALUE + offset, mask=valid[:, None], other=0).to(tl.float32)
    # Native sigmoid materializes a BF16 tensor; omitting this rounding
    # would silently change the operation even if the final dtype is BF16.
    rounded = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(OUTPUT + offset, rounded[:, None] * value, mask=valid[:, None])


def sigmoid(values):
    if values.dtype != torch.bfloat16 or not values.is_contiguous():
        raise ValueError('Only contiguous BF16 gates admitted')
    output = torch.empty_like(values)
    _sigmoid[(triton.cdiv(values.numel(), 256),)](values, output, SIZE=values.numel(), BLOCK=256)
    return output


def gate_multiply(gate, values, block_rows=2):
    if (values.ndim != 2 or values.shape[1] != 2048 or not 0 < values.shape[0] <= 208
            or gate.shape != (values.shape[0], 1) or gate.dtype != values.dtype
            or values.dtype != torch.bfloat16 or gate.device != values.device
            or not values.is_contiguous() or not gate.is_contiguous()
            or block_rows not in (1, 2, 4)):
        raise ValueError('Pinned BF16 token-by2048 and token-by1 geometry required')
    output = torch.empty_like(values)
    _gate_multiply[(triton.cdiv(values.shape[0], block_rows),)](gate, values, output,
        TOKENS=values.shape[0], BLOCK_ROWS=block_rows)
    return output
