"""CPU-only bounds for the private grouped FIA experiment."""

from bisect import bisect_right
from dataclasses import dataclass

MAX_WIDTH = 13
PAGE_SIZE = 128


def attention_graph_key(context, tokens):
    """Use exactly the framework graph identity, never just token count.

    Dynamic gamma can produce distinct FULL graphs with identical token
    counts, e.g.35=5x7=7x5. Their handles, queries and graph-pool diagnostic
    buffers must not be updated or inspected through the other graph.
    """
    descriptor = context.batch_descriptor
    if descriptor is None or descriptor.num_tokens != tokens:
        raise ValueError("Attention graph requires its exact batch descriptor")
    hash(descriptor)  # Fail closed if a caller supplies mutable graph state.
    return descriptor


@dataclass(frozen=True)
class MaskedLayout:
    tokens: int
    width: int
    groups: int
    upper_lengths: tuple[int, ...]


def masked_layout(query_ends, upper_lengths, actual_reqs, max_context):
    """Keep uniform requests grouped; use width1 for irregular partitions.

    Only CPU structural query lengths and conservative CPU KV bounds enter
    this function. Exact causal limits are supplied separately on device.
    """
    if not query_ends or not 0 < actual_reqs <= len(query_ends):
        raise ValueError("Missing request/query bounds")
    starts = (0, *map(int, query_ends))
    lengths = tuple(b - a for a, b in zip(starts, starts[1:]))
    if min(lengths) <= 0 or len(upper_lengths) < actual_reqs:
        raise ValueError("Invalid cumulative query lengths or KV bounds")
    if max_context <= 0 or max_context % PAGE_SIZE:
        raise ValueError("Context must contain complete physical pages")
    for length, upper in zip(lengths[:actual_reqs], upper_lengths[:actual_reqs]):
        if not length <= int(upper) <= max_context:
            raise ValueError("A CPU KV upper bound does not cover its query")
    tokens = starts[-1]
    first = lengths[0]
    uniform = first <= MAX_WIDTH and all(v == first for v in lengths[:actual_reqs])
    width = first if uniform else 1
    # Any additional query rows represent the runner's one dummy request.
    # No grouped query may straddle two real requests.
    groups = (tokens + width - 1) // width
    bounds = []
    for group in range(groups):
        request = bisect_right(starts, group * width) - 1
        if request >= actual_reqs:
            bounds.append(1)
        else:
            upper = int(upper_lengths[request])
            bounds.append(min(max_context, ((upper + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE))
    return MaskedLayout(tokens, width, groups, tuple(bounds))
