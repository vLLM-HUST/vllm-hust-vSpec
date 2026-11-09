"""Clip a NEW verified output before native count production; never mutate it."""
import triton
import triton.language as tl

@triton.jit
def clip_kernel(source,output,counters,OLD:tl.constexpr,NEW:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    column=tl.arange(0,BLOCK)
    value=tl.load(source+row*(OLD+1)+column,column<OLD+1,other=-1)
    value=tl.where(column<NEW+1,value,-1)
    tl.store(output+row*(OLD+1)+column,value,column<OLD+1)
    if row==0:tl.atomic_add(counters+OLD,1)

def clip_verified_prefix(source,old,new,counters):
    import torch
    if old not in (4,6,8) or new!=old-2:
        raise ValueError('Adjacent even shrink only')
    if (source.ndim!=2 or not 1<=source.shape[0]<=16 or source.shape[1]!=old+1
            or source.dtype!=torch.int32 or not source.is_contiguous()
            or source.device.type!='npu' or counters.shape!=(9,)
            or counters.dtype!=torch.int32 or counters.device!=source.device or not counters.is_contiguous()):
        raise ValueError('Original int32 padded sampler geometry required')
    output=torch.empty_like(source)
    clip_kernel[(source.shape[0],)](source,output,counters,OLD=old,NEW=new,BLOCK=16)
    return output
