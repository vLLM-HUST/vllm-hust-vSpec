"""Device geometry for experimental exact-masked grouped FIA.

No device-to-host reads. Request bounds and exclusive partial-page ownership
must be established by the coordinator before these kernels are admitted.
"""

import triton
import triton.language as tl


@triton.jit
def exact_mask_geometry(seq_lens, qsl, original_table, control, mask, safe_table,
                        MAX_CONTEXT: tl.constexpr, TABLE_COLS: tl.constexpr,
                        TABLE_STRIDE: tl.constexpr, MAX_REQS: tl.constexpr,
                        ROW_CAPACITY: tl.constexpr, PAGE: tl.constexpr,
                        BLOCK: tl.constexpr):
    token = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = tl.load(control)
    tokens = tl.load(control + 1)
    nreqs = tl.load(control + 2)
    groups = (tokens + width - 1) // width
    rows = groups * width
    requests = tl.arange(0, MAX_REQS)
    starts = tl.load(qsl + requests, mask=requests < nreqs, other=2147483647)
    req = tl.sum(((token >= starts) & (requests < nreqs)).to(tl.int32), 0) - 1
    req_end = tl.load(qsl + req + 1, mask=(req >= 0) & (req < nreqs), other=0)
    req_start = tl.load(qsl + req, mask=(req >= 0) & (req < nreqs), other=0)
    live = (token < tokens) & (token < req_end) & (req >= 0)
    seq = tl.load(seq_lens + req, mask=live, other=1)
    causal_end = tl.where(live, seq - (req_end - req_start) + token - req_start + 1, 1)
    tl.store(mask + token * MAX_CONTEXT + offset, offset >= causal_end,
             mask=(token < rows) & (token < ROW_CAPACITY) & (offset < MAX_CONTEXT))
    # For grouped queries every live row belongs to the same request.
    # A width1 group is also valid for arbitrary query partitions.
    physical = tl.load(original_table + req * TABLE_STRIDE + offset,
                       mask=live & (offset < TABLE_COLS) & (offset < (seq + PAGE - 1) // PAGE), other=0)
    tl.store(safe_table + (token // width) * TABLE_COLS + offset, physical,
             mask=(token % width == 0) & (token < rows) & (offset < TABLE_COLS))


@triton.jit
def zero_invisible_tails(key, value, seq_lens, original_table, control,
                         PAGE: tl.constexpr, PLANE_DIM: tl.constexpr,
                         TABLE_STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    # The null physical block0 is reserved by the pool. Every actual last
    # partial page is exclusively owned: prefix caching only shares full pages.
    req = tl.program_id(0) - 1
    lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    nreqs = tl.load(control + 2)
    live = (req >= 0) & (req < nreqs)
    # Keep scalar GM addresses valid even for inactive/null programs. Some
    # Ascend scalar lowering paths evaluate a masked scalar load's address.
    safe_req = tl.minimum(tl.maximum(req, 0), nreqs - 1)
    seq = tl.load(seq_lens + safe_req)
    column = tl.minimum(seq // PAGE, TABLE_STRIDE - 1)
    loaded = tl.load(original_table + safe_req * TABLE_STRIDE + column)
    physical = tl.where(live & (seq % PAGE != 0), loaded, 0)
    zero = (req == -1) | (live & (seq % PAGE != 0) & (lane // PLANE_DIM >= seq % PAGE))
    index = physical * PAGE * PLANE_DIM + lane
    tl.store(key + index, 0, mask=zero & (lane < PAGE * PLANE_DIM))
    tl.store(value + index, 0, mask=zero & (lane < PAGE * PLANE_DIM))


@triton.jit
def grouped_query_pack(source, packed, control, TOKENS: tl.constexpr,
                       HEADS: tl.constexpr, DIM: tl.constexpr,
                       CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    width = tl.load(control)
    groups = (TOKENS + width - 1) // width
    active = lane < groups * HEADS * width * DIM
    dim = lane % DIM
    position = lane // DIM % width
    head = lane // (DIM * width) % HEADS
    request = lane // (DIM * width * HEADS)
    token = request * width + position
    value = tl.load(source + (token * HEADS + head) * DIM + dim,
                    mask=active & (token < TOKENS), other=0)
    tl.store(packed + lane, value, mask=active & (lane < CAPACITY))


@triton.jit
def grouped_query_unpack(packed, output, control, TOKENS: tl.constexpr,
                         HEADS: tl.constexpr, DIM: tl.constexpr,
                         BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = lane < TOKENS * HEADS * DIM
    width = tl.load(control)
    dim = lane % DIM
    head = lane // DIM % HEADS
    token = lane // (DIM * HEADS)
    index = ((token // width * HEADS + head) * width + token % width) * DIM + dim
    value = tl.load(packed + index, mask=active, other=0)
    tl.store(output + lane, value, mask=active)
