"""Three-group stable-buffer refresh for private Qwen3.5 experiments.

Validated on both allocated 910B2 devices, including full-buffer equality and
gamma4/gamma6 graph replay. The runner coordinator performs admission and can
compare against the native GDN builders; no sampling/model math is changed.

Admission must prove all row/column bounds before launch.  The kernel does not
mask an invalid source column into a plausible zero, and callers must never
fall back after launch starts.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_reqs", "token_count"])
def gdn_gid123_full_stable_refresh(
    table1,
    table2,
    table3,
    seq_lens,
    query_start_loc,
    accepted,
    state1,
    masks1,
    tokens1,
    query1,
    accepted1,
    actual1,
    state2,
    masks2,
    tokens2,
    query2,
    accepted2,
    actual2,
    state3,
    masks3,
    tokens3,
    query3,
    accepted3,
    actual3,
    num_reqs,
    token_count,
    TABLE_COLS: tl.constexpr,
    TABLE_STRIDE0: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Refresh three independent builder storages in one program grid."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

    state_lane = lane < STATE_CAPACITY * STATE_WIDTH
    state_row = lane // STATE_WIDTH
    state_col = lane % STATE_WIDTH
    live_state = state_lane & (state_row < num_reqs)

    seq_len = tl.load(seq_lens + state_row, mask=live_state, other=0)
    start_col = tl.maximum((seq_len - 1) // MAMBA_BLOCK_SIZE, 0)
    source_col = start_col + state_col

    # Bounds are deliberately prelaunch obligations.  For an admitted frame,
    # state_row is in [0, num_reqs) and source_col is in [0, TABLE_COLS).
    value1 = tl.load(
        table1 + state_row * TABLE_STRIDE0 + source_col,
        mask=live_state,
        other=0,
    )
    value2 = tl.load(
        table2 + state_row * TABLE_STRIDE0 + source_col,
        mask=live_state,
        other=0,
    )
    value3 = tl.load(
        table3 + state_row * TABLE_STRIDE0 + source_col,
        mask=live_state,
        other=0,
    )
    tl.store(state1 + lane, value1, mask=state_lane)
    tl.store(state2 + lane, value2, mask=state_lane)
    tl.store(state3 + lane, value3, mask=state_lane)

    request_lane = lane < num_reqs
    true_values = tl.full([BLOCK], 1, tl.int1)
    tl.store(masks1 + lane, true_values, mask=request_lane)
    tl.store(masks2 + lane, true_values, mask=request_lane)
    tl.store(masks3 + lane, true_values, mask=request_lane)

    accepted_value = tl.load(accepted + lane, mask=request_lane, other=0)
    tl.store(accepted1 + lane, accepted_value, mask=request_lane)
    tl.store(accepted2 + lane, accepted_value, mask=request_lane)
    tl.store(accepted3 + lane, accepted_value, mask=request_lane)

    token_lane = lane < token_count
    tl.store(tokens1 + lane, lane, mask=token_lane)
    tl.store(tokens2 + lane, lane, mask=token_lane)
    tl.store(tokens3 + lane, lane, mask=token_lane)

    query_lane = lane <= num_reqs
    query_value = tl.load(query_start_loc + lane, mask=query_lane, other=0)
    previous_query = tl.load(
        query_start_loc + lane - 1,
        mask=query_lane & (lane > 0),
        other=0,
    )
    actual_value = tl.where(lane == 0, query_value, query_value - previous_query)
    tl.store(query1 + lane, query_value, mask=query_lane)
    tl.store(query2 + lane, query_value, mask=query_lane)
    tl.store(query3 + lane, query_value, mask=query_lane)
    tl.store(actual1 + lane, actual_value, mask=query_lane)
    tl.store(actual2 + lane, actual_value, mask=query_lane)
    tl.store(actual3 + lane, actual_value, mask=query_lane)


# Coordinator grid:
#   (triton.cdiv(STATE_CAPACITY * STATE_WIDTH, BLOCK),)
# STATE_WIDTH is five or seven, matching the adaptive runner's current builder
# storage (not merely the draft-token count, which may lag during transitions).
# The centralized environment switch defaults to off.
