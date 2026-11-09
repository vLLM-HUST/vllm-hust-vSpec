"""Exact gather/predicate/multiply fusion; IDs are native top8 outputs0..255."""

import torch
import triton
import triton.language as tl


@triton.jit
def expert_mask(IDS, WEIGHTS, EXPERT_MAP, OUTPUT, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = index < COUNT
    expert = tl.load(IDS + index, mask=live, other=0)
    mapped = tl.load(EXPERT_MAP + expert, mask=live, other=-1)
    weight = tl.load(WEIGHTS + index, mask=live, other=0).to(tl.float32)
    result = weight * (mapped != -1).to(tl.float32)
    tl.store(OUTPUT + index, result, mask=live)


def fused_mask(ids, weights, expert_map):
    if (ids.ndim != 2 or ids.shape[1] != 8 or weights.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype not in (torch.bfloat16, torch.float32)
            or expert_map.dtype != torch.int32 or expert_map.numel() != 256
            or not all(value.is_contiguous() for value in (ids, weights, expert_map))):
        raise ValueError('Only contiguous native top8 IDs,256experts,FP32/BF16weights are admitted')
    output = torch.empty_like(weights)
    expert_mask[(triton.cdiv(ids.numel(), 256),)](ids, weights, expert_map, output, ids.numel(), BLOCK=256)
    return output
