"""Parallel target FIA task rebinding for EAGLE FULL graphs."""

from __future__ import annotations


def apply_parallel_graph_update_patch(workers: int) -> bool:
    """Parallelize EAGLE target graph updates with the shared ABI adapter.

    Qwen3.5 target models mix GDN and standard attention layers. The shared
    updater filters graph metadata down to captured FIA layers and handles
    both the legacy and current Ascend capture tuple layouts. EAGLE draft
    updates remain on the host path because its one-layer graph has too little
    independent work to split profitably.
    """
    if workers <= 1:
        return False

    from .draft_parallel_update import apply_draft_parallel_graph_update_patch

    return apply_draft_parallel_graph_update_patch(
        workers=0,
        target_workers=workers,
    )
