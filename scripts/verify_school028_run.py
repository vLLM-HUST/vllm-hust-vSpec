"""Verify original-load MMLU service evidence, not task accuracy."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(root, run):
    directory = root / run
    receipt = read(directory / "receipt.json")
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert not receipt.get("startup_foreign_context_observations")
    assert receipt["child_exit_codes"] in ([0, 0], [0, 130])
    log = (directory / "service.log").read_text()
    if receipt["child_exit_codes"][-1] == 130:
        assert "request processing complete; starting resource teardown" in log
    for name, digest in receipt["input_hashes"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    cache = read(directory / "compiler-cache-initialization.json")
    assert cache["seed_manifest_sha256"] == receipt["input_hashes"]["microflow-cache-seed.json"]
    assert cache["vllm_aot_cache"] == "fresh"
    raw = read(directory / "mmlu-pro.json")
    assert raw["completed"] == 200 and raw["failed"] == 0
    assert raw["total_output_tokens"] == 51200 and raw["total_input_tokens"] == 282265
    assert len(raw["output_lens"]) == 200 and set(raw["output_lens"]) == {256}
    control = read(root / "mmlu-b1-kvmat-e24-r36-screen-r1/mmlu-pro.json")
    assert raw["input_lens"] == control["input_lens"]
    command = read(directory / "launch-command.json")
    for flag, value in {
        "--max-num-seqs": "24",
        "--max-model-len": "65536",
        "--max-num-batched-tokens": "4096",
        "--kv-cache-memory-bytes": "21474836480",
        "--tensor-parallel-size": "2",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--enable-prefix-caching" in command and "--async-scheduling" in command
    assert json.loads(command[command.index("--speculative-config") + 1]) == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    policy = events(directory / "kvmat-policy.jsonl")
    admitted = [
        e
        for e in events(directory / "kvmat-engine.jsonl")
        if e.get("event") == "school028_request_salt_admitted"
    ]
    assert len(policy) == len(admitted) == 200
    assert all(e["runtime_effective_decision"] == "recompute" for e in policy)
    assert {hashlib.sha256(e["runtime_cache_salt"].encode()).hexdigest() for e in policy} == {
        e["cache_salt_sha256"] for e in admitted
    }
    active = receipt["measurement_role"].endswith(" on")
    start, end = receipt["bench_started_wall"], receipt["bench_ended_wall"]
    micro_ranks, ranks = [], []
    for path in directory.glob("microflow-*.jsonl"):
        rows = events(path)
        loaded = [e for e in rows if e["event"] == "model_loaded"]
        if not loaded:
            continue
        assert len(loaded) == 1 and loaded[0]["active"]
        checks = [e for e in rows if e["event"] == "partition_equivalence"]
        assert {e["tokens"] for e in checks} == {1024, 2048, 4096}
        assert all(all(e["exact"]) for e in checks)
        captures = [e for e in rows if e["event"] == "prefix" and e["capturing"] and e["split"]]
        assert len({e["layer"] for e in captures}) == 10
        assert any(
            e["event"] == "graph_replay" and e["target"] and start <= e["time"] <= end for e in rows
        )
        micro_ranks.append(loaded[0]["rank"])
    for path in directory.glob("vspec-*.jsonl"):
        rows = events(path)
        loaded = [e for e in rows if e["event"] == "model_loaded"]
        if not loaded:
            continue
        assert len(loaded) == 1 and loaded[0]["active"] == active
        checks = [e for e in rows if e["event"] == "head_equivalence"]
        assert {(e["rows"], e["kind"]) for e in checks} == {
            (n, k) for n in (1, 24, 36, 48) for k in ("zeros", "sin")
        }
        assert all(e["exact"] for e in checks)
        captures = [
            e
            for e in rows
            if e["event"] == "selector_executed" and e.get("capturing") and e["active"] == active
        ]
        assert captures
        replays = [
            e
            for e in rows
            if e["event"] == "draft_graph_replay"
            and e["active"] == active
            and start <= e["time"] <= end
        ]
        assert replays
        ranks.append(dict(rank=loaded[0]["rank"], draft_replays=len(replays)))
    assert set(micro_ranks) == {0, 1} and {e["rank"] for e in ranks} == {0, 1}
    assert abs(raw["total_output_tokens"] / raw["duration"] - raw["output_throughput"]) < 1e-8
    return dict(
        run_id=run,
        active=active,
        requests=200,
        failures=0,
        ranks=ranks,
        quality_rescored=False,
        source_commit=receipt["vspec_source_commit"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("run")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.run), indent=2))
