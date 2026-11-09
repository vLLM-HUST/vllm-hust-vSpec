"""Exact native-top8 consumer fusion; no router score/selection/weight change.

Contract: IDs from the unchanged native top8 producer, hence0..255. This is
not a general replacement for Python advanced-index negative/OOB semantics.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _mask(IDS,WEIGHTS,EXPERT_MAP,OUTPUT,COUNTERS,
          COUNT:tl.constexpr,REQUESTS:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    live=index<COUNT
    expert=tl.load(IDS+index,mask=live,other=0)
    mapped=tl.load(EXPERT_MAP+expert,mask=live,other=-1)
    weight=tl.load(WEIGHTS+index,mask=live,other=0).to(tl.float32)
    result=weight*(mapped!=-1).to(tl.float32)
    tl.store(OUTPUT+index,result,mask=live)
    if tl.program_id(0)==0:
        tl.atomic_add(COUNTERS+REQUESTS,1)

def fused(ids,weights,expert_map,counts):
    if (ids.ndim!=2 or ids.shape[1]!=8 or ids.shape[0]%7!=0
            or not 7<=ids.shape[0]<=112 or weights.shape!=ids.shape
            or ids.dtype not in (torch.int32,torch.int64)
            or weights.dtype not in (torch.bfloat16,torch.float32)
            or expert_map.shape!=(256,) or expert_map.dtype!=torch.int32
            or counts.shape!=(17,) or counts.dtype!=torch.int32
            or not all(x.is_contiguous() and x.device==ids.device for x in (ids,weights,expert_map,counts))):
        raise ValueError('Native top8 IDs0..255, width7 N1..16, BF16/FP32 weights,256entry map,17cell counter required')
    output=torch.empty_like(weights)
    _mask[(triton.cdiv(ids.numel(),256),)](ids,weights,expert_map,output,counts,
        COUNT=ids.numel(),REQUESTS=ids.shape[0]//7,BLOCK=256)
    return output
