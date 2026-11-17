"""Real checkpoint head layout: exact BF16 outputs on every admitted shape."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SHAPES = dict(draft=tuple(range(1, 17)),
    target=tuple(sorted(set(range(1, 17)) | {n * q for n in range(1, 17) for q in (5, 7)} | {208})))
SCALES = (0.125, 1., 2., 8., 0.01, 0.5, 16., 1.)


def main():
    import torch
    import torch_npu
    from kernel import execute
    from real_weight import load_weights
    os.umask(0o077)
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank); torch.set_num_threads(1); torch.manual_seed(311030 + rank)
    torch.npu.config.allow_internal_format = True
    device = torch.device('npu', rank); weights, provenance = load_weights(rank); records = []
    for role, cpu_weight in weights.items():
        native = cpu_weight.to(device)
        packed = torch_npu.npu_format_cast(native if role == 'draft' else native.t().contiguous(), 29)
        candidate = packed if role == 'draft' else packed.t()
        assert torch.equal(candidate.cpu().contiguous().view(torch.int16), cpu_weight.view(torch.int16))
        for rows in SHAPES[role]:
            x = torch.randn((rows, 2048), dtype=torch.bfloat16, device=device)
            graphs = {}
            for arm, weight in (('native', native), ('candidate', candidate)):
                for _ in range(3): execute(x, weight)
                torch.npu.synchronize(); graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph): held = execute(x, weight)
                graphs[arm] = graph, held
            checks = []
            for revision, scale in enumerate(SCALES):
                value = (torch.randn((rows, 2048)) * scale).bfloat16()
                if revision == 7:
                    value[0].zero_(); value[0, ::2] = -0.0
                x.copy_(value.to(device)); expected = execute(x, native); torch.npu.synchronize()
                check = dict(revision=revision, scale=scale, arms={})
                for arm, (graph, held) in graphs.items():
                    graph.replay(); torch.npu.synchronize()
                    bits = int((held.cpu().contiguous().view(torch.int16) != expected.cpu().contiguous().view(torch.int16)).sum())
                    nonfinite = int((~torch.isfinite(held) | ~torch.isfinite(expected)).sum().cpu())
                    check['arms'][arm] = dict(bit_mismatches=bits, nonfinite=nonfinite, exact=bits == nonfinite == 0,
                        output_format=torch_npu.get_npu_format(held))
                assert check['arms']['native']['exact'], 'Native graph/reference mismatch'
                assert torch.equal(x.cpu().view(torch.int16), value.view(torch.int16))
                checks.append(check)
            timing = {}
            if (role, rows) in (('draft', 14), ('target', 70), ('target', 98)):
                samples = {arm: [] for arm in graphs}
                for repetition in range(5):
                    for arm in (('native', 'candidate') if repetition % 2 == 0 else ('candidate', 'native')):
                        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                        begin.record()
                        for _ in range(30): graphs[arm][0].replay()
                        end.record(); end.synchronize()
                        samples[arm].append(begin.elapsed_time(end) * 1000 / 30)
                timing = {arm: statistics.median(v) for arm, v in samples.items()}
            record = dict(role=role, rows=rows, vocab=cpu_weight.shape[0], checks=checks,
                exact=all(c['arms']['candidate']['exact'] for c in checks),
                graph_median_us=timing, actual_weight_bits_unchanged=True,
                weight_format=torch_npu.get_npu_format(candidate), weight_stride=list(candidate.stride()))
            records.append(record)
            print(json.dumps(dict(rank=rank, role=role, rows=rows, exact=record['exact'], timing=timing)), flush=True)
            del graphs, expected, x
    paths = [Path(__file__), ROOT / 'kernel.py', ROOT / 'real_weight.py']
    output = args.output.with_name(args.output.stem + f'-rank{rank}.json')
    with output.open('x') as stream:
        json.dump(dict(completed=True, rank=rank, records=records, provenance=provenance,
            real_checkpoint_weights=True, runtime_internal_format_enabled=True, full_model_native_and_quality_still_required=True,
            goal_complete=False, sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), stream, indent=2)


if __name__ == '__main__': main()
