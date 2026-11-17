"""Two immutable ND BF16 shards; fused gather and native positive-zero semantics."""
import torch
import triton
import triton.language as tl


def geometry(ids,local,remote,rank):
    return bool(type(rank) is int and rank in (0,1) and ids.ndim==1
        and ids.dtype in (torch.int32,torch.int64) and local.ndim==remote.ndim==2
        and local.shape==remote.shape and local.shape[0]>0 and local.shape[1]>0
        and local.dtype==remote.dtype==torch.bfloat16
        and ids.is_contiguous() and local.is_contiguous() and remote.is_contiguous()
        and ids.device==local.device==remote.device)


@triton.jit(do_not_specialize=['ROWS'])
def _split_lookup(I,L,R,O,C,ROWS,HALF:tl.constexpr,WIDTH:tl.constexpr,
                  RANK:tl.constexpr,BLOCK:tl.constexpr,COUNT:tl.constexpr):
    row=tl.program_id(0)
    tile=tl.program_id(1)
    col=tile*BLOCK+tl.arange(0,BLOCK)
    valid=(row<ROWS) & (col<WIDTH)
    pos=row*WIDTH+col
    token=tl.load(I+row,row<ROWS,other=0).to(tl.int32)
    owner=tl.where(token<HALF,0,1)
    index=tl.where(token<HALF,token,token-HALF)
    token_valid=valid & (token>=0) & (token<2*HALF)
    left=tl.load(L+index*WIDTH+col,token_valid & (owner==RANK),other=0)
    right=tl.load(R+index*WIDTH+col,token_valid & (owner!=RANK),other=0)
    value=tl.where(owner==RANK,left,right)
    # Native TP embedding sums its owner plus positive BF16 zero. Preserve
    # nonzero bits; explicitly canonicalize the negative-zero encoding.
    bits=value.to(tl.int16,bitcast=True)
    bits=tl.where(bits==-32768,0,bits).to(tl.int16)
    tl.store(O+pos,bits.to(tl.bfloat16,bitcast=True),valid)
    if COUNT:
        if (row==0) & (tile==0):
            tl.atomic_add(C+ROWS,1)


def embedding(ids,local,remote,rank,block=512,counter=None):
    if not geometry(ids,local,remote,rank) or block not in (512,1024,2048):
        raise ValueError('Pinned contiguous same-device BF16 two-bank embedding geometry required')
    output=torch.empty((ids.numel(),local.shape[1]),device=local.device,dtype=local.dtype)
    if counter is not None and (counter.dtype!=torch.int32 or counter.device!=ids.device
            or counter.ndim!=1 or counter.numel()<=ids.numel()):
        raise ValueError('Mutable same-device int32 counter must cover row index')
    if ids.numel():
        _split_lookup[(ids.numel(),triton.cdiv(local.shape[1],block))](
            ids,local,remote,output,counter,ids.numel(),local.shape[0],local.shape[1],rank,block,counter is not None)
    return output
