"""Private Q35 TP2 GDN input fusion: split, L2 Q/K, value copy and gates.

Target recurrence, weights, state updates and output projection are untouched.
Admit only after both-device exact numerical comparisons, including canaries.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['TOKENS'])
def fused_q35_gdn_inputs(MIXED, A, B, A_LOG, DT_BIAS, PACKED, G, BETA,
                         TOKENS, EPS: tl.constexpr, BLOCK_TOKENS: tl.constexpr):
    token = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    qk_column = tl.arange(0, 1024)
    value_column = tl.arange(0, 2048)
    gate_column = tl.arange(0, 16)
    valid = token < TOKENS
    # All GM transfers are affine rectangular slices. In particular, avoid
    # vector div/mod and interleaved head scatter in global pointer offsets.
    offset = token[:, None] * 4096 + qk_column[None, :]
    q = tl.load(MIXED + offset, mask=valid[:, None], other=0).to(tl.float32).reshape((BLOCK_TOKENS * 8, 128))
    k = tl.load(MIXED + offset + 1024, mask=valid[:, None], other=0).to(tl.float32).reshape((BLOCK_TOKENS * 8, 128))
    q_square = q * q
    k_square = k * k
    q_sum = tl.sum(q_square, 1)[:, None]
    k_sum = tl.sum(k_square, 1)[:, None]
    q_rsqrt = tl.rsqrt(q_sum + EPS)
    k_rsqrt = tl.rsqrt(k_sum + EPS)
    destination = token[:, None] * 1024 + qk_column[None, :]
    tl.store(PACKED + destination, (q * q_rsqrt).reshape((BLOCK_TOKENS, 1024)), mask=valid[:, None])
    tl.store(PACKED + TOKENS * 1024 + destination, (k * k_rsqrt).reshape((BLOCK_TOKENS, 1024)), mask=valid[:, None])
    value = tl.load(MIXED + token[:, None] * 4096 + 2048 + value_column[None, :], mask=valid[:, None], other=0)
    tl.store(PACKED + TOKENS * 2048 + token[:, None] * 2048 + value_column[None, :], value, mask=valid[:, None])
    gate_offset = token[:, None] * 16 + gate_column[None, :]
    a = tl.load(A + gate_offset, mask=valid[:, None], other=0).to(tl.float32)
    b = tl.load(B + gate_offset, mask=valid[:, None], other=0).to(tl.float32)
    alog = tl.load(A_LOG + gate_column).to(tl.float32)
    bias = tl.load(DT_BIAS + gate_column).to(tl.float32)
    x = a + bias[None, :]
    softplus = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    gate = -tl.exp(alog)[None, :] * softplus
    tl.store(G + gate_offset, gate, mask=valid[:, None])
    tl.store(BETA + gate_offset, tl.sigmoid(b), mask=valid[:, None])


def fused_inputs(mixed, a, b, a_log, dt_bias, block_tokens=4):
    if (mixed.ndim != 2 or mixed.shape[1] != 4096 or mixed.dtype != torch.bfloat16
            or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or a.shape != b.shape or a.shape != (mixed.shape[0], 16)
            or a_log.numel() != 16 or dt_bias.numel() != 16
            or not all(value.is_contiguous() for value in (mixed, a, b, a_log, dt_bias))):
        raise ValueError('Private fused GDN inputs require BF16-only contiguous Q35 TP2 geometry')
    tokens = mixed.shape[0]
    packed = torch.empty(tokens * 4096, dtype=mixed.dtype, device=mixed.device)
    g = torch.empty((1, tokens, 16), dtype=torch.float32, device=mixed.device)
    beta = torch.empty((1, tokens, 16), dtype=b.dtype, device=mixed.device)
    fused_q35_gdn_inputs[(triton.cdiv(tokens, block_tokens),)](
        mixed, a, b, a_log, dt_bias, packed, g, beta,
        TOKENS=tokens, EPS=1e-6, BLOCK_TOKENS=block_tokens)
    q = packed[:tokens * 1024].view(1, tokens, 8, 128)
    k = packed[tokens * 1024:tokens * 2048].view(1, tokens, 8, 128)
    v = packed[tokens * 2048:].view(1, tokens, 16, 128)
    return q, k, v, g, beta
