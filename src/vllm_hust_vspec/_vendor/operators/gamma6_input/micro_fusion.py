"""Compare exact native GDN input operators and private fusion on one NPU."""

import argparse
import json
from pathlib import Path
import statistics
TOKENS=tuple(n*7 for n in range(1,17))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    import torch
    import torch_npu  # noqa: F401
    import triton
    from kernel import fused_inputs, fused_q35_gdn_inputs
    from reference import native_inputs
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    init_device_properties_triton()
    device, records = f'npu:{args.device}', []
    generator = torch.Generator().manual_seed(12763)
    for tokens, bias_dtype in [(tokens, dtype) for tokens in TOKENS
                               for dtype in (torch.bfloat16, torch.float32)]:
        # Actual unquantized BF16 projection path only. FP32 a/b are rejected
        # by the callable; no change to BF16 beta or FP32 g tolerances.
        for gate_dtype in (torch.bfloat16,):
            mixed_backing = torch.full((tokens * 4096 + 32,), -77, dtype=torch.bfloat16, device=device)
            mixed = mixed_backing[16:-16].view(tokens, 4096)
            a = torch.empty((tokens, 16), dtype=gate_dtype, device=device)
            b = torch.empty_like(a)
            a_log = torch.empty(16, dtype=torch.float32, device=device)
            dt_bias = torch.empty(16, dtype=bias_dtype, device=device)
            packed_backing = torch.full((tokens * 4096 + 32,), -79, dtype=torch.bfloat16, device=device)
            g_backing = torch.full((tokens * 16 + 32,), -91, dtype=torch.float32, device=device)
            beta_backing = torch.full((tokens * 16 + 32,), -83, dtype=gate_dtype, device=device)
            packed, g, beta = packed_backing[16:-16], g_backing[16:-16], beta_backing[16:-16]
            mixed.fill_(1)
            a.fill_(0)
            b.fill_(0)
            a_log.fill_(0)
            dt_bias.fill_(0)
            native_inputs(mixed, a, b, a_log, dt_bias)
            fused_inputs(mixed, a, b, a_log, dt_bias)
            torch.npu.synchronize()
            graphs, outputs = {}, {}
            for name, function in [('native', native_inputs), ('fused', fused_inputs)]:
                graphs[name] = torch.npu.NPUGraph()
                with torch.npu.graph(graphs[name]):
                    outputs[name] = function(mixed, a, b, a_log, dt_bias)
            guarded_graph = torch.npu.NPUGraph()
            with torch.npu.graph(guarded_graph):
                fused_q35_gdn_inputs[(triton.cdiv(tokens, 4),)](
                    mixed, a, b, a_log, dt_bias, packed, g, beta,
                    TOKENS=tokens, EPS=1e-6, BLOCK_TOKENS=4)
            for phase, scale in enumerate((1.0, 0.0, 2.0**-20, 256.0)):
                host_mixed = torch.randn((tokens, 4096), generator=generator).mul_(scale).to(torch.bfloat16)
                host_a = torch.randn((tokens, 16), generator=generator).to(gate_dtype)
                host_b = torch.randn((tokens, 16), generator=generator).to(gate_dtype)
                if phase == 3:
                    host_a[:, ::2] = 30
                    host_a[:, 1::2] = -30
                    host_b[:, ::2] = 100
                    host_b[:, 1::2] = -100
                mixed.copy_(host_mixed.to(device))
                a.copy_(host_a.to(device))
                b.copy_(host_b.to(device))
                a_log.copy_(torch.randn(16, generator=generator).to(device))
                dt_bias.copy_(torch.randn(16, generator=generator).to(device))
                immutable=(mixed,a,b,a_log,dt_bias)
                before=[t.clone() for t in immutable]
                for _ in range(3):
                    graphs['native'].replay()
                    graphs['fused'].replay()
                    guarded_graph.replay()
                torch.npu.synchronize()
                native, actual = [value.cpu() for value in outputs['native']], [value.cpu() for value in outputs['fused']]
                eager=[value.cpu() for value in native_inputs(mixed,a,b,a_log,dt_bias)]
                for left,right in zip(native,eager):
                    assert torch.equal(left.view(torch.int16 if left.dtype==torch.bfloat16 else torch.int32),
                                       right.view(torch.int16 if right.dtype==torch.bfloat16 else torch.int32)), 'Original native self-graph control failed'
                fused_eager=[value.cpu() for value in fused_inputs(mixed,a,b,a_log,dt_bias)]
                for left,right in zip(actual,fused_eager):
                    dtype=torch.int16 if left.dtype==torch.bfloat16 else torch.int32
                    assert torch.equal(left.view(dtype),right.view(dtype)), 'Fused eager/self-graph mismatch'
                differences = [(left.float() - right.float()).abs().max().item() for left, right in zip(native, actual)]
                assert all(torch.equal(native[index], actual[index]) for index in (0, 1, 2, 4)), differences
                assert all(torch.equal(native[index].view(torch.int16),actual[index].view(torch.int16)) for index in (0,1,2,4)), 'BF16 consumed bits including signed zero differ'
                torch.testing.assert_close(native[3], actual[3], rtol=1e-6, atol=1e-6)
                expected_packed = torch.cat([actual[index].flatten() for index in (0, 1, 2)])
                assert torch.equal(packed.cpu(), expected_packed)
                assert torch.equal(g.cpu(), actual[3].flatten())
                assert torch.equal(beta.cpu(), actual[4].flatten())
                assert torch.equal(mixed.cpu(), host_mixed)
                for left,right in zip(immutable,before):
                    dtype=torch.int16 if left.dtype==torch.bfloat16 else torch.int32
                    assert torch.equal(left.view(dtype),right.view(dtype)), 'Complete input modified'
                for tensor, sentinel in [(mixed_backing, -77), (packed_backing, -79), (g_backing, -91), (beta_backing, -83)]:
                    assert bool((tensor[:16].cpu() == sentinel).all())
                    assert bool((tensor[-16:].cpu() == sentinel).all())
                records.append(dict(tokens=tokens, gate_dtype=str(gate_dtype), phase=phase,
                    bias_dtype=str(bias_dtype),
                    scale=scale, max_abs=differences,
                    q_k_value_beta_bitexact=True, fp32_gate_rtol=1e-6, fp32_gate_atol=1e-6,
                    canaries_unchanged=True, inputs_unchanged=True, replays=3))
                records[-1]['original_native_eager_self_graph_bits_exact']=True
                records[-1]['actual_consumed_BF16_bits_including_signed_zero_exact']=True
                records[-1]['fused_eager_self_graph_bits_exact']=True
                records[-1]['complete_all_five_inputs_raw_bits_readonly']=True
            paired = {'native': [], 'fused': []}
            for round_index in range(8):
                order = ('native', 'fused') if round_index % 2 == 0 else ('fused', 'native')
                for name in order:
                    start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    paired[name].append(start.elapsed_time(end) / 30)
            records.append(dict(tokens=tokens, gate_dtype=str(gate_dtype),
                                bias_dtype=str(bias_dtype),
                                paired_median_ms={name: statistics.median(values) for name, values in paired.items()},
                                paired_samples_ms=paired))
        print('[fused-gdn-input-micro] ' + json.dumps(dict(tokens=tokens, bias_dtype=str(bias_dtype), cases=len(records))), flush=True)
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, device=args.device, cases=records, full_model_validated=False,
                       end_to_end_performance_claim=False, accuracy_evaluated=False,
                       native_qk_norm_value_beta_bitexact=True, fp32_gate_rtol=1e-6, fp32_gate_atol=1e-6), stream, indent=2)


if __name__ == '__main__':
    main()
