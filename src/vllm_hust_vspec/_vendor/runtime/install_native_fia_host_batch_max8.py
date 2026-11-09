"""Private host-only boxing install: no twin graph or original event changes."""
from functools import wraps
import os


def install():
    from vllm_ascend.worker.worker import NPUWorker
    from .native_fia_host_batch_max8 import NativeFIAHostBatch
    original = NPUWorker.load_model
    if getattr(original, '_private_max8_host_boxing', False):
        return

    @wraps(original)
    def load_model(worker, *args, **kwargs):
        result = original(worker, *args, **kwargs)
        assert os.environ['VLLM_ASCEND_Q35_TWIN_GRAPH'] == 'off'
        assert os.environ['VLLM_ASCEND_Q35_REPLAY_TAIL'] == 'validate'
        runner = worker.model_runner
        prior = runner._q35_native_fia_host_batch
        assert prior.mode == 'off' and not prior.enabled and not prior.entries
        helper = NativeFIAHostBatch(runner, 'on')
        assert helper.mode == 'on' and helper.enabled
        assert runner.speculative_config.num_speculative_tokens == 8
        runner._q35_native_fia_host_batch = helper
        return result

    load_model._private_max8_host_boxing = True
    NPUWorker.load_model = load_model
