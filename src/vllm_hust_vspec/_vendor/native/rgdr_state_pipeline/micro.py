"""Unchanged native recurrence versus buffered writes: all state slots exact."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
VARIANTS = dict(base128=(128, 1), serial96=(96, 1), pipe96=(96, 2),
                serial80=(80, 1), pipe80=(80, 2), serial64=(64, 1), pipe64=(64, 2))


def main():
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    os.umask(0o077)
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank); torch.set_num_threads(1)
    torch.ops.load_library(str(ROOT / 'build-r1/lib/libvspec_rgdr_adapter.so'))
    native_op = torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    candidate_op = torch.ops.vspec_rgdr_pipeline.project
    records = []
    for width in (5, 7):
        tokens = 14 * width
        for seed in range(24):
            torch.manual_seed(1031 + seed)
            q = torch.nn.functional.normalize(torch.randn(tokens, 8, 128, device='npu'), dim=-1).to(torch.bfloat16)
            k = torch.nn.functional.normalize(torch.randn(tokens, 8, 128, device='npu'), dim=-1).to(torch.bfloat16)
            v = torch.randn(tokens, 16, 128, device='npu', dtype=torch.bfloat16) / 8
            g = -torch.rand(tokens, 16, device='npu', dtype=torch.float32) / 4
            beta = torch.rand(tokens, 16, device='npu', dtype=torch.bfloat16)
            initial = torch.randn(128, 16, 128, 128, device='npu', dtype=torch.float32) / 8
            initial[127].fill_(float('nan'))
            states = {arm: initial.clone() for arm in ('native', *VARIANTS)}
            offset = (0, 7, 13, 29)[seed % 4]
            indices = (torch.randperm(tokens, device='npu') + offset).to(torch.int32)
            route = ('uniform', 'uneven', 'padded_zero')[seed // 8]
            lens = ([width] * 14 if route == 'uniform' else
                    [width - 1, width + 1] * 7 if route == 'uneven' else
                    [0] + [8] * 7 + ([4] + [2] * 5 if width == 5 else [7] * 6))
            assert sum(lens) == tokens and len(lens) == 14 and max(lens) <= 8
            lengths = torch.tensor([0] + lens, device='npu', dtype=torch.int32)
            accepted = torch.tensor([0 if n == 0 else 1 + (i + seed) % n for i, n in enumerate(lens)], device='npu', dtype=torch.int32)

            def execute(arm):
                if arm == 'native':
                    return native_op(query=q, key=k, value=v, state=states[arm], beta=beta, g=g,
                        scale=128 ** -.5, actual_seq_lengths=lengths, ssm_state_indices=indices,
                        num_accepted_tokens=accepted)
                tile, buffers = VARIANTS[arm]
                return candidate_op(q, k, v, beta, states[arm], lengths, indices, g, accepted, tile, buffers)

            def check(arm, actual, expected, phase):
                torch.npu.synchronize()
                assert bool(torch.isfinite(actual).all().cpu()) and bool(torch.isfinite(expected).all().cpu())
                assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (width, seed, arm, phase, 'output')
                assert torch.equal(states[arm].view(torch.int32), states['native'].view(torch.int32)), (width, seed, arm, phase, 'all intermediate and unused states')

            print(json.dumps(dict(stage='case', rank=rank, width=width, seed=seed)), flush=True)
            expected = execute('native'); torch.npu.synchronize()
            for arm in VARIANTS:
                check(arm, execute(arm), expected, 'eager')
            graphs = {}
            for arm in states:
                for _ in range(3):
                    states[arm].copy_(initial); execute(arm)
                torch.npu.synchronize(); graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph): held = execute(arm)
                graphs[arm] = graph, held
            for replay in range(8):
                for state in states.values(): state.copy_(initial)
                accepted.copy_(torch.tensor([0 if n == 0 else 1 + (i + seed + replay) % n for i, n in enumerate(lens)],
                                           dtype=torch.int32, device='npu'))
                expected = execute('native'); torch.npu.synchronize()
                # Native capture must agree with a separately completed eager
                # from the same reset bank, never with another broken capture.
                expected_state = states['native'].clone()
                states['native'].copy_(initial); graphs['native'][0].replay(); torch.npu.synchronize()
                assert torch.equal(states['native'].view(torch.int32), expected_state.view(torch.int32))
                check('native', graphs['native'][1], expected, f'self_graph{replay}')
                for arm in VARIANTS:
                    graphs[arm][0].replay()
                    check(arm, graphs[arm][1], expected, f'graph{replay}')
            samples = {arm: [] for arm in graphs}; names = list(graphs)
            for repetition in range(5):
                order = names[repetition % len(names):] + names[:repetition % len(names)]
                for arm in order:
                    states[arm].copy_(initial)
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(24): graphs[arm][0].replay()
                    end.record(); end.synchronize()
                    samples[arm].append(begin.elapsed_time(end) * 1000 / 24)
            record = dict(width=width, seed=seed, route=route, all_eager_and_eight_graph_outputs_and_entire_state_bits_exact=True,
                native_eager_and_graph_self_reference_passed=True, unused_NaN_payload_slot127_preserved=True,
                median_us={arm: statistics.median(v) for arm, v in samples.items()})
            records.append(record); print(json.dumps(dict(rank=rank, **record)), flush=True)
            del graphs, graph, held, expected, expected_state, initial
            states.clear()
            torch.npu.empty_cache()
    paths = [ROOT / name for name in ('micro.py', 'adapter.cpp', 'kernel.cpp', 'rgdr_full128.h',
        'recurrent_gated_delta_rule_tiling_data.h', 'CMakeLists.txt',
        'build-r1/lib/libvspec_rgdr_adapter.so', 'build-r1/lib/libvspec_rgdr_kernel.so')]
    output = args.output.with_name(args.output.stem + f'-rank{rank}.json')
    with output.open('x') as stream:
        json.dump(dict(completed=True, rank=rank, records=records,
            component_only_no_model_or_performance_or_quality_admission=True,
            no_FP32_state_or_BF16_input_precision_change=True,
            every_native_state_write_preserved=True, goal_complete=False,
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), stream, indent=2)


if __name__ == '__main__': main()
