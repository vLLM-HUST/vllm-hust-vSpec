"""Exact integer gather/add for admitted Q35 async target preparation."""

import triton
import triton.language as tl


@triton.jit
def exact_target_positions(computed, requests, query_positions, output,
                           TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = index < TOKENS
    request = tl.load(requests + index, mask=live, other=0)
    base = tl.load(computed + request, mask=live, other=0).to(tl.int64)
    offset = tl.load(query_positions + index, mask=live, other=0).to(tl.int64)
    tl.store(output + index, base + offset, mask=live)


@triton.jit(do_not_specialize=['TOKENS'])
def runtime_target_positions(computed, requests, query_positions, output,
                             TOKENS, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = index < TOKENS
    request = tl.load(requests + index, mask=live, other=0)
    base = tl.load(computed + request, mask=live, other=0).to(tl.int64)
    offset = tl.load(query_positions + index, mask=live, other=0).to(tl.int64)
    tl.store(output + index, base + offset, mask=live)
