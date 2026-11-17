"""Immutable native FP32 weights only; no router algorithm/event changes."""
import functools
import torch
import torch.nn.functional as F


def key(value):
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (value.data_ptr(), value.storage_offset(), tuple(value.shape),
            tuple(value.stride()), value.dtype, value.device, version)


class FrozenWeight:
    def __init__(self, gate, original):
        self.gate, self.source, self.original = gate, gate.weight, original
        self.source_key = key(self.source)
        self.precast = getattr(gate, 'weight_fp32', None)
        self.weight = original(gate)
        self.weight_key = key(self.weight)
        self.calls = 0

    def get(self):
        if (self.gate.weight is not self.source or key(self.source) != self.source_key
                or getattr(self.gate, 'weight_fp32', None) is not self.precast
                or key(self.weight) != self.weight_key):
            raise RuntimeError('Immutable native FP32 router weight changed')
        self.calls += 1  # Python invocation/capture witness, NOT device replay count.
        return self.weight


def install():
    from vllm_ascend.ops.fused_moe import q35_immutable_cache as native
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    import torch_npu
    original, load = native.native_weight, NPUModelRunner.load_model
    records = {}

    def weight(gate):
        record = records.get(id(gate))
        return original(gate) if record is None else record.get()

    @functools.wraps(load)
    def load_then_bind(runner, *args, **kwargs):
        result = load(runner, *args, **kwargs)
        assert not records
        # Existing independent admission checks Q35, BF16, TP2, immutable scope.
        assert runner._q35_native_lookup_manager.mode == 'on'
        assert len(runner._q35_native_lookup_manager.helpers) == 11
        assert runner._q35_immutable_moe_manager.mode == 'off'
        receipts = []
        for role, model in (('target', runner.model), ('draft', runner.drafter.model)):
            for name, owner in model.named_modules():
                if not isinstance(owner, AscendMoERunner):
                    continue
                assert not owner._q35_single_stream_events
                assert getattr(owner, '_q35_immutable_cache', None) is None
                gate = owner.gate
                assert gate is not None and id(gate) not in records
                assert gate.weight.dtype == torch.bfloat16 and tuple(gate.weight.shape) == (256, 2048)
                assert gate.weight.is_contiguous() and torch_npu.get_npu_format(gate.weight) in (0, 2)
                record = FrozenWeight(gate, original)
                assert record.weight.dtype == torch.float32 and record.weight.is_contiguous()
                assert torch_npu.get_npu_format(record.weight) in (0, 2)
                reference = original(gate)
                assert torch.equal(record.weight.view(torch.int32), reference.view(torch.int32))
                counts = [1, 14, 42, 70, 98, 126, 208]
                for n in counts:
                    # Both calls retain original native FP32 input and F.linear.
                    values = ((torch.arange(n * 2048, device=record.weight.device,
                        dtype=torch.float32).reshape(n, 2048) % 59 - 29) / 17).bfloat16().float()
                    actual = F.linear(values, record.weight)
                    expected = F.linear(values, original(gate))
                    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                records[id(gate)] = record
                receipts.append(dict(role=role, layer=name, rows=counts,
                    complete_native_FP32_weight_bits_exact=True,
                    original_native_FP32_linear_bits_exact=True,
                    original_precast_present=record.precast is not None,
                    original_MoE_events_and_shared_gate_UNCHANGED=True))
        assert len(records) == 41
        assert sum(row['role'] == 'target' for row in receipts) == 40
        assert sum(row['role'] == 'draft' for row in receipts) == 1
        runner._private_frozen_router_records = records
        runner._private_frozen_router_receipts = receipts
        print('[private-router-weight-cache] original_native_FP32_linear=True layers=41 load_time_bits_exact=True original_events_retained=True', flush=True)
        return result

    native.native_weight = weight
    NPUModelRunner.load_model = load_then_bind
