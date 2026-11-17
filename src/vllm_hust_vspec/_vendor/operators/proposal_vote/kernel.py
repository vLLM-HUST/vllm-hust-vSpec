"""Exact TP2 BF16 local-max voting; projection and native collective unchanged.

Only the gather dimension/layout changes: rank-major rather than row-major.
NaNs win over numbers and ordinary ties retain rank0. Actual Ascend argmax
prefers positive zero over negative zero; CPU argmax does NOT, so preserve
the measured NPU convention explicitly rather than borrowing CPU semantics.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'])
def _pack(V, I, IDS, OUT, N, B: tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    index=tl.load(I+row,row<N,other=0)
    token=tl.load(IDS+index,row<N,other=0)
    value=tl.load(V+row,row<N,other=0).to(tl.float32)
    tl.store(OUT+2*row,value,row<N)
    tl.store(OUT+2*row+1,token.to(tl.float32),row<N)


@triton.jit(do_not_specialize=['N'])
def _unpack(P, OUT, C, N, B: tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    v0=tl.load(P+2*row,row<N,other=0)
    v1=tl.load(P+2*(N+row),row<N,other=0)
    t0=tl.load(P+2*row+1,row<N,other=0)
    t1=tl.load(P+2*(N+row)+1,row<N,other=0)
    # Native NPU argmax: first NaN, first ordinary maximum, +0 over -0.
    zero_precedence=(v0==0) & (v1==0) & (v0.to(tl.int32,bitcast=True)<0) & (v1.to(tl.int32,bitcast=True)>=0)
    choose1=(v1>v0) | ((v1!=v1) & (v0==v0)) | zero_precedence
    token=tl.where(choose1,t1,t0).to(tl.int64)
    tl.store(OUT+row,token,row<N)
    if tl.program_id(0)==0:
        tl.atomic_add(C+N,1)


def pack(values,indices,ids):
    n=values.numel()
    if (values.ndim!=1 or values.dtype not in (torch.bfloat16,torch.float32)
            or indices.shape!=values.shape or indices.dtype!=torch.int64
            or ids.ndim!=1 or ids.dtype!=torch.int64 or not 1<=n<=8192
            or not all(x.is_contiguous() for x in (values,indices,ids))
            or values.device!=indices.device or values.device!=ids.device):
        raise ValueError('Require contiguous same-device BF16/FP32 values and int64 IDs/indices, N1..8192')
    output=torch.empty((n,2),device=values.device,dtype=torch.float32)
    _pack[(triton.cdiv(n,128),)](values,indices,ids,output,n,128)
    return output


def unpack(gathered,counter):
    n=gathered.shape[0]//2
    if (gathered.ndim!=2 or gathered.shape!=(n*2,2) or not 1<=n<=8192
            or gathered.dtype!=torch.float32 or not gathered.is_contiguous()
            or counter.ndim!=1 or counter.numel()!=8193 or counter.dtype!=torch.int32
            or counter.device!=gathered.device):
        raise ValueError('Require rank-major TP2 FP32 pairs and ordinary mutable int32[8193] counter')
    output=torch.empty(n,device=gathered.device,dtype=torch.int64)
    _unpack[(triton.cdiv(n,128),)](gathered,output,counter,n,128)
    return output
