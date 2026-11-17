"""Native eager/captured attention oracle for original per-group update order."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'build-r1/lib'))
import vspec_fia_host
from guard_controls import check as check_guard


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard_controls = check_guard(vspec_fia_host)
    torch.npu.set_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(10312026)
    keys = [torch.randn(224, 128, 256, dtype=torch.bfloat16, device='npu') / 8 for _ in range(10)]
    values = [torch.randn_like(key) for key in keys]
    tables = [torch.arange(224, dtype=torch.int32, device='npu').reshape(14, 16).roll(i, 0)
              for i in range(10)]
    mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device='npu'), diagonal=1)
    update_stream = torch.npu.Stream()
    records, comparisons = [], 0
    plans = [(requests, role, gamma, widths) for requests in (1, 2, 3, 14)
             for role, gamma, widths in [('target', 4, [5] * 10), ('target', 6, [7] * 10),
             ('draft', 4, [5, 1, 1, 1]), ('draft', 6, [7, 1, 1, 1, 1, 1])]]
    for requests, role, gamma, widths in plans:
        width, count = gamma + 1, len(widths)
        queries = [torch.zeros(requests * width, 8, 256, dtype=torch.bfloat16, device='npu') for _ in widths]
        for q, live_width in zip(queries, widths):
            q[:requests * live_width].copy_(torch.randn(requests * live_width, 8, 256, dtype=torch.bfloat16, device='npu') / 4)
        qs = [list(range(1, requests * width + 1)) if role == 'draft' and layer > 0 else
              [w * (i + 1) for i in range(requests)] for layer, w in enumerate(widths)]
        lengths = [[512 + 128 * (i % 8) + (layer if role == 'draft' else 0) for i in range(requests)]
                   for layer in range(count)]
        for layer in range(count):
            if role == 'draft' and layer > 0:
                lengths[layer].extend([0] * (requests * width - requests))
        task_tables = [torch.cat((t[:requests], torch.zeros((requests * width - requests, 16), dtype=t.dtype, device=t.device)), 0) if role == 'draft' and layer > 0 else t[:requests]
                       for layer, t in enumerate(tables[:count])]
        base = dict(atten_mask=mask,
                    input_layout='TND', block_size=128, num_key_value_heads=1, num_heads=8,
                    scale=256 ** -.5, sparse_mode=3, pre_tokens=2147483647, next_tokens=2147483647)
        workspaces = [torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query=queries[layer], key=keys[layer], value=values[layer], block_table=task_tables[layer],
            actual_seq_lengths=qs[layer], actual_seq_lengths_kv=lengths[layer], **base)
            for layer in range(count)]
        workspace = max(workspaces, key=lambda value: value.numel())
        outputs = {arm: [torch.full_like(q, 1 / 64) for q in queries] for arm in ('native', 'candidate')}
        lses = {arm: [torch.empty(1, dtype=torch.bfloat16, device='npu') for _ in queries]
                for arm in ('native', 'candidate')}
        bindings = [vspec_fia_host.NativeFIA(q, keys[layer], values[layer], mask, workspace, out, lse,
                    8, 1, 128, 3, 2147483647, 2147483647, 256 ** -.5)
                    for layer, (q, out, lse) in enumerate(zip(queries, outputs['candidate'], lses['candidate']))]

        def native(arm, layer, current, current_tables, capture=False):
            torch_npu.npu_fused_infer_attention_score.out(query=queries[layer],
                key=keys[layer], value=values[layer], block_table=current_tables[layer],
                actual_seq_lengths=qs[layer],
                actual_seq_lengths_kv=current[layer], workspace=workspace,
                out=[outputs[arm][layer], lses[arm][layer]], **base)

        def compare(tag):
            nonlocal comparisons
            torch.npu.synchronize()
            for left, right, live_width in zip(outputs['native'], outputs['candidate'], widths):
                live = requests * live_width
                assert torch.isfinite(left[:live]).all() and torch.isfinite(right[:live]).all(), (width, tag, 'nonfinite')
                assert torch.equal(left[:live].view(torch.int16), right[:live].view(torch.int16)), (width, tag, 'consumed output bits')
                comparisons += 1

        for layer in range(count):
            native('native', layer, lengths, task_tables)
            bindings[layer].invoke(qs[layer], lengths[layer], task_tables[layer])
        compare('eager')
        graphs, groups = {}, {}
        for arm in ('native', 'candidate'):
            for layer in range(count):
                native(arm, layer, lengths, task_tables)
            torch.npu.synchronize()
            graph, events, handles = torch.npu.NPUGraph(), [], []
            with torch.npu.graph(graph):
                stream = torch.npu.current_stream()
                for layer in range(count):
                    event = torch.npu.ExternalEvent()
                    event.wait(stream)
                    event.reset(stream)
                    torch.npu.graph_task_group_begin(stream)
                    native(arm, layer, lengths, task_tables, capture=True)
                    handles.append(torch.npu.graph_task_group_end(stream))
                    events.append(event)
            graphs[arm], groups[arm] = graph, (events, handles)
        batch = vspec_fia_host.NativeFIABatch(bindings, groups['candidate'][1], groups['candidate'][0])

        def update(arm, current, current_tables):
            with torch.npu.stream(update_stream):
                if arm == 'candidate':
                    batch.update(update_stream, qs, current, current_tables[:count])
                else:
                    events, handles = groups[arm]
                    for layer in range(count):
                        torch.npu.graph_task_update_begin(update_stream, handles[layer])
                        native(arm, layer, current, current_tables)
                        torch.npu.graph_task_update_end(update_stream)
                        events[layer].record(update_stream)

        for replay in range(8):
            current = [[n + (replay + 1) * (1 + i % 7) if i < requests else 0
                        for i, n in enumerate(row)] for row in lengths]
            current_tables = [t.roll(replay % 3, 1).clone() for t in task_tables]
            for arm in ('native', 'candidate'):
                for out, live_width in zip(outputs[arm], widths):
                    if requests * live_width < out.shape[0]:
                        out[requests * live_width:].fill_(1 / 64)
                graphs[arm].replay()
                update(arm, current, current_tables)
                torch.npu.synchronize()
            compare(f'graph{replay}')
        times = {'native': [], 'candidate': []}
        for repeat in range(5):
            for arm in (('native', 'candidate') if repeat % 2 == 0 else ('candidate', 'native')):
                current = [[n + 64 + repeat if i < requests else 0 for i, n in enumerate(row)] for row in lengths]
                torch.npu.synchronize()
                graphs[arm].replay()
                start = time.perf_counter()
                update(arm, current, task_tables)
                times[arm].append((time.perf_counter() - start) * 1000)
                torch.npu.synchronize()
        record = dict(role=role, gamma=gamma, widths=widths, width=width, requests=requests, native_groups=count,
                      native_update_host_ms=statistics.median(times['native']),
                      candidate_update_host_ms=statistics.median(times['candidate']), raw_timing_ms=times,
                      output_comparisons=9 * count, all_8_replays_use_changed_exact_host_lengths=True,
                      per_task_distinct_KV_and_replaced_block_table_tensors=True,
                      captured_query_rows_each_task=[q.shape[0] for q in queries],
                      consumed_output_rows_each_task=[requests * w for w in widths],
                      capture_query_lengths_full_outer_then_runtime_exact_live_prefix=True,
                      original_per_row_padding_after_real_requests_preserved=True,
                      per_task_actual_query_lengths=qs,
                      unused_padding_outputs_NOT_counted_as_semantic_rows=True)
        records.append(record)
        print(json.dumps(record), flush=True)
        del graphs, graph, groups, batch, bindings
    assert comparisons == 1080
    paths = [ROOT / p for p in ('micro.py', 'guard_controls.py', 'adapter.cpp', 'CMakeLists.txt', 'build-r1/lib/vspec_fia_host.so')]
    with args.output.open('x') as stream:
        json.dump(dict(passed=True, device=args.device, compiled_guard_controls=guard_controls, comparisons=comparisons, records=records,
                       same_native_FIA_and_original_per_group_event_order=True,
                       primitive_only=True, actual_compact_native_FIA_padding_geometry_exercised=True,
                       model_integration_NOT_implemented=True, goal_complete=False,
                       sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), stream, indent=2)


if __name__ == '__main__':
    main()
