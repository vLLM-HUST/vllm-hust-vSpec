"""Install private worker RPC without modifying the sealed Ascend backend."""

from ..config import q35_tail_prefix_shrink_enabled


def install_prefix_shrink_rpc():
    if not q35_tail_prefix_shrink_enabled():
        return False
    from vllm_ascend.worker.worker import NPUWorker
    current = getattr(NPUWorker, 'fence_private_q35_prefix_shrink', None)
    if current is not None:
        if not getattr(current, '_private_tail3_prefix_rpc', False):
            raise RuntimeError('Conflicting prefix-shrink RPC')
        return False

    def fence_private_q35_prefix_shrink(worker, transition_id, expected_roster):
        if not q35_tail_prefix_shrink_enabled():
            raise RuntimeError('Private tail3 prefix shrink is disabled')
        from .q35_prefix_shrink_fence import prepare_prefix_shrink
        return prepare_prefix_shrink(worker.model_runner, transition_id, expected_roster)

    fence_private_q35_prefix_shrink._private_tail3_prefix_rpc = True
    NPUWorker.fence_private_q35_prefix_shrink = fence_private_q35_prefix_shrink
    return True
