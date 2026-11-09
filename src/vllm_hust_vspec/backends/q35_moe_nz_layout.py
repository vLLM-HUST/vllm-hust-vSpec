"""Default-off, load-time-only, value-exact expert-weight layout experiment.

No inference method or arithmetic is wrapped. This screen is not a complete
runtime native-operator proof; it must not be admitted from layout bits alone.
"""
from functools import wraps
import json
import os


def convert_layer(layer):
    import torch
    import torch_npu

    if hasattr(layer, '_q35_moe_nz_receipt'):
        raise RuntimeError('Duplicate NZ expert transform')
    records = []
    for name, expected in (('w13_weight', (128, 2048, 1024)), ('w2_weight', (128, 512, 2048))):
        parameter = getattr(layer, name)
        source = parameter.data
        if (tuple(source.shape) != expected or source.dtype != torch.bfloat16
                or not source.is_contiguous() or torch_npu.get_npu_format(source) not in (0, 2)):
            raise RuntimeError('Pinned BF16 ND expert geometry changed')
        version = parameter._version
        transformed = torch_npu.npu_format_cast(source, 29)
        recovered = torch_npu.npu_format_cast(transformed, 2)
        # All consumed weights, all bits, before any graph capture or timer.
        bad = ((source.view(torch.int16) != recovered.view(torch.int16))
               | ~torch.isfinite(source) | ~torch.isfinite(recovered)).any()
        if bool(bad.cpu()) or parameter._version != version:
            raise RuntimeError('NZ conversion changed original BF16 expert values')
        if torch_npu.get_npu_format(transformed) != 29:
            raise RuntimeError('Actual NZ weight format missing')
        parameter.data = transformed
        records.append(dict(name=name, shape=list(expected), original_version=version,
                            actual_format=29, complete_BF16_weight_bits_exact=True))
    receipt = dict(pid=os.getpid(), weights=records, inference_methods_unchanged=True,
                   no_numeric_precision_or_quantization_change=True,
                   load_time_only_no_formal_runtime_native_or_quality_admission=True)
    layer._q35_moe_nz_receipt = receipt
    print('[q35-moe-nz-load] ' + json.dumps(receipt), flush=True)


def install_moe_nz_layout():
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

    original = AscendUnquantizedFusedMoEMethod.process_weights_after_loading
    if getattr(original, '_q35_moe_nz_layout', False):
        return

    @wraps(original)
    def process_weights_after_loading(self, layer):
        original(self, layer)
        convert_layer(layer)

    process_weights_after_loading._q35_moe_nz_layout = True
    AscendUnquantizedFusedMoEMethod.process_weights_after_loading = process_weights_after_loading

    from vllm_ascend.worker.worker import NPUWorker

    original_metrics = NPUWorker.get_private_q35_gdn_input_metrics

    @wraps(original_metrics)
    def metrics(self):
        import torch_npu

        records = original_metrics(self)
        layouts = []
        runner = self.model_runner
        for role, model in (('target', runner.model), ('draft', runner.drafter.model)):
            for name, layer in model.named_modules():
                receipt = getattr(layer, '_q35_moe_nz_receipt', None)
                if receipt is None:
                    continue
                for weight in receipt['weights']:
                    parameter = getattr(layer, weight['name'])
                    if (torch_npu.get_npu_format(parameter) != 29
                            or list(parameter.shape) != weight['shape']):
                        raise RuntimeError('Actual expert layout changed after load')
                layouts.append(dict(role=role, layer=name, **receipt))
        if ([row['role'] for row in layouts].count('target') != 40
                or [row['role'] for row in layouts].count('draft') != 1):
            raise RuntimeError('Require forty target and one draft expert banks')
        if not records:
            raise RuntimeError('Missing original GDN metrics')
        records[0]['moe_nz_layout'] = layouts
        return records

    metrics._q35_moe_nz_layout = True
    NPUWorker.get_private_q35_gdn_input_metrics = metrics
