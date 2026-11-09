"""Native arithmetic with explicit physical input row stride; no contiguous copy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from itertools import product

ROOT = Path(__file__).resolve().parent
OLD = Path(os.environ['VSPEC_CONV_REFERENCE_DIR'])
TILES = (256, 512, 1024, 1376, 1536, 2048, 4096)
ROWS = (1, 8, 14, 16)
ROUTES = ('uniform', 'uneven', 'padded_zero')
TYPES = ('i32', 'i64', 'mixed', 'wide_indices', 'capacity8_indices')
LAYOUTS = ('contiguous', 'qkvz_split_6144')
STATE_LENGTHS = (9, 11)


def main():
    import torch
    import torch_npu  # noqa: F401
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    torch.set_num_threads(1)
    torch.manual_seed(10314251 + rank)
    torch.ops.load_library(str(OLD / 'build-r1/lib/libvspec_conv_tile_adapter.so'))
    torch.ops.load_library(str(ROOT / 'build-r1/lib/libvspec_conv_stride_adapter.so'))
    records = []
    for rows in ROWS:
        for width in (5, 7):
            tokens = rows * width
            for route in ROUTES:
                for types, layout, state_len in product(TYPES, LAYOUTS, STATE_LENGTHS):
                    print(json.dumps(dict(stage='case', rank=rank, rows=rows, width=width,
                                          route=route, types=types, layout=layout, state_len=state_len)), flush=True)
                    integer = torch.int64 if types in ('i64', 'mixed') else torch.int32
                    index_integer = torch.int32 if types != 'i64' else torch.int64
                    columns = 4096 if layout == 'contiguous' else 6144
                    value_base = torch.empty((tokens, columns), dtype=torch.bfloat16, device='npu')
                    value = value_base[:, :4096]
                    if layout == 'qkvz_split_6144':
                        value_base[:, 4096:].view(torch.int16).fill_(32705)
                        value_base[:, -1].view(torch.int16).fill_(-32768)
                        assert value.stride() == (6144, 1) and not value.is_contiguous()
                    else:
                        assert value.is_contiguous()
                    weight = (torch.randn((4, 4096), device='npu') / 16).to(torch.bfloat16)
                    initial = (torch.randn((192, state_len, 4096), device='npu') / 16).to(torch.bfloat16)
                    initial.view(torch.int16)[191].fill_(32705)
                    initial.view(torch.int16)[190, -1, -1] = -32768
                    lens = [width] * rows
                    if route == 'uneven':
                        lens = [width - int(i % 2 == 0) for i in range(rows)]
                    elif route == 'padded_zero':
                        lens[0] = 0
                    prefix = [0]
                    for length in lens:
                        prefix.append(prefix[-1] + length)
                    queries = torch.tensor(prefix, dtype=integer, device='npu')
                    stride = 9 if types == 'capacity8_indices' else (7 if types == 'wide_indices' else width)
                    indices = (torch.arange(rows * stride, dtype=index_integer, device='npu')
                               .reshape(rows, stride).flip(0).contiguous())
                    accepted = torch.empty(rows, dtype=integer, device='npu')
                    copy_arms = ('copy2048', 'copy4096') if state_len == 9 and stride <= 7 else ()
                    arms = ('native', *copy_arms, *map(str, TILES))
                    states = {a: initial.clone() for a in arms}
                    outputs = {a: torch.empty_like(value) for a in arms}
                    counters = {a: torch.zeros(34, dtype=torch.int32, device='npu') for a in arms}

                    def reset(arm, states=states, outputs=outputs, counters=counters, initial=initial):
                        states[arm].copy_(initial)
                        outputs[arm].view(torch.int16).fill_(32705)
                        counters[arm].zero_()

                    def invoke(arm, states=states, outputs=outputs, value=value, weight=weight,
                               queries=queries, indices=indices, accepted=accepted, counters=counters):
                        if arm == 'native':
                            torch.ops._C_ascend.npu_causal_conv1d_custom(outputs[arm], value, weight,
                                conv_state=states[arm], bias_opt=None, query_start_loc_opt=queries,
                                cache_indices_opt=indices, initial_state_mode_opt=None,
                                num_accepted_tokens_opt=accepted, activation_mode=1, pad_slot_id=-1, run_mode=1)
                        elif arm.startswith('copy'):
                            torch.ops.vspec_conv_tile.project(outputs[arm], value.contiguous(), weight, states[arm],
                                                              queries, indices, accepted, counters[arm], int(arm[4:]))
                        else:
                            torch.ops.vspec_conv_strided.project(outputs[arm], value, weight, states[arm],
                                                              queries, indices, accepted, counters[arm], int(arm))

                    def check(arm, expected, expected_state, phase, states=states, outputs=outputs,
                              counters=counters, rows=rows, width=width):
                        assert torch.equal(outputs[arm].view(torch.int16), expected.view(torch.int16)), (rows, width, route, types, arm, phase, 'full output')
                        assert torch.equal(states[arm].view(torch.int16), expected_state.view(torch.int16)), (rows, width, route, types, arm, phase, 'all physical state bits')
                        wanted = torch.zeros_like(counters[arm])
                        if arm != 'native':
                            wanted[rows + (17 if width == 7 else 0)] = 1
                        assert torch.equal(counters[arm], wanted), (rows, width, arm, phase, 'all replay witness cells')

                    value.normal_(0, .0625)
                    accepted.fill_(1)
                    graphs = {}
                    for arm in arms:
                        for _ in range(3):
                            reset(arm)
                            invoke(arm)
                        torch.npu.synchronize()
                        reset(arm)
                        graph = torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            invoke(arm)
                        graphs[arm] = graph
                    for revision in range(8):
                        value.normal_(0, (.0625, .5, 1, 4, .01, 8, .125, 2)[revision])
                        accepted.copy_(torch.tensor([0 if n == 0 else 1 + (i + revision) % n
                                                     for i, n in enumerate(lens)], dtype=integer, device='npu'))
                        sources = [t.clone() for t in (value_base, weight, queries, indices, accepted)]
                        reset('native')
                        invoke('native')
                        torch.npu.synchronize()
                        expected, expected_state = outputs['native'].clone(), states['native'].clone()
                        reset('native')
                        graphs['native'].replay()
                        torch.npu.synchronize()
                        check('native', expected, expected_state, 'independent native selfgraph')
                        for arm in arms[1:]:
                            reset(arm)
                            invoke(arm)
                            torch.npu.synchronize()
                            check(arm, expected, expected_state, f'eager{revision}')
                            reset(arm)
                            graphs[arm].replay()
                            torch.npu.synchronize()
                            check(arm, expected, expected_state, f'graph{revision}')
                        assert all(torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                                               b.view(torch.int16) if b.dtype == torch.bfloat16 else b)
                                   for a, b in zip((value_base, weight, queries, indices, accepted), sources))
                    medians = {}
                    if route == 'uniform' and types in ('i32', 'capacity8_indices'):
                        samples = {a: [] for a in arms}
                        for repetition in range(7):
                            order = arms[repetition % len(arms):] + arms[:repetition % len(arms)]
                            for arm in order:
                                reset(arm)
                                begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                                begin.record()
                                for _ in range(40):
                                    graphs[arm].replay()
                                end.record()
                                end.synchronize()
                                samples[arm].append(begin.elapsed_time(end) * 1000 / 40)
                        medians = {a: statistics.median(v) for a, v in samples.items()}
                    record = dict(rows=rows, width=width, route=route, types=types, layout=layout,
                        state_len=state_len, index_stride=stride, old_copy_control_arms=list(copy_arms),
                        value_stride=list(value.stride()),
                        candidate_uses_original_physical_input_stride_without_contiguous_copy=True,
                        old_copy2048_and_copy4096_controls_include_their_contiguous_copy=True,
                        entire_qkvz_source_base_including_unused_z_NaN_and_signed_zero_immutable=True,
                        all_eager_and_eight_graph_outputs_and_all_physical_state_bits_exact=True,
                        independent_original_native_selfgraph_passed=True,
                        unused_output_padding_and_unused_state_NaN_and_signed_zero_bits_checked=True,
                        source_value_weight_and_metadata_unchanged=True, median_us=medians)
                    record['actual_replay_witness_folded_inside_same_kernel_all_cells_checked'] = True
                    records.append(record)
                    print(json.dumps(dict(rank=rank, **record)), flush=True)
                    del graphs, graph, expected, expected_state, initial, sources
                    states.clear()
                    outputs.clear()
                    torch.npu.empty_cache()
    paths = [ROOT / name for name in ('micro.py', 'adapter.cpp', 'kernel.cpp', 'CMakeLists.txt',
        'build-r1/lib/libvspec_conv_stride_adapter.so', 'build-r1/lib/libvspec_conv_stride_kernel.so')]
    paths += list((ROOT / 'native-kernel').rglob('*.h'))
    output = args.output.with_name(args.output.stem + f'-rank{rank}.json')
    with output.open('x') as stream:
        json.dump(dict(completed=True, rank=rank, records=records,
            original_native_arithmetic_events_output_and_state_writeback_unchanged=True,
            only_five_input_address_calculations_and_private_stride_ABI_changed=True,
            component_only_no_model_or_quality_or_performance_admission=True, goal_complete=False,
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), stream, indent=2)


if __name__ == '__main__':
    main()
