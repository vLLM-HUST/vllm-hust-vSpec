from __future__ import annotations

from vllm_hust_vspec.online_benchmark import (
    build_benchmark_command,
    parse_args,
    resolve_tokenizer,
)


def test_arc_easy_draft_benchmark_matches_reference_contract() -> None:
    options = parse_args(
        [
            "--method",
            "draft",
            "--vllm-executable",
            "/usr/local/python3.11.14/bin/vllm",
            "--tokenizer",
            "/model/Qwen2.5-14B-Instruct",
        ]
    )
    command = build_benchmark_command(options)

    assert command[:3] == [
        "/usr/local/python3.11.14/bin/vllm",
        "bench",
        "serve",
    ]
    assert command[command.index("--backend") + 1] == "openai-chat"
    assert command[command.index("--base-url") + 1] == "http://127.0.0.1:18180"
    assert command[command.index("--endpoint") + 1] == "/v1/chat/completions"
    assert command[command.index("--model") + 1] == "qwen2.5-14b-draft-vspec-fp16"
    assert command[command.index("--tokenizer") + 1] == "/model/Qwen2.5-14B-Instruct"
    assert command[command.index("--dataset-name") + 1] == "custom"
    assert command[command.index("--dataset-path") + 1] == "/run_dir/materialized.jsonl"
    assert "--disable-shuffle" in command
    assert command[command.index("--custom-output-len") + 1] == "256"
    assert command[command.index("--num-prompts") + 1] == "200"
    assert command[command.index("--request-rate") + 1] == "inf"
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--seed") + 1] == "0"
    assert command[command.index("--result-dir") + 1].endswith("ARC-Easy-draft-vspec")
    assert command[command.index("--result-filename") + 1] == "ARC-Easy.json"


def test_arc_easy_eagle_defaults_and_overrides() -> None:
    options = parse_args(
        [
            "--method",
            "eagle",
            "--served-model-name",
            "custom-name",
            "--result-dir",
            "/tmp/result",
            "--vllm-executable",
            "/usr/bin/vllm",
            "--",
            "--percentile-metrics",
            "ttft",
        ]
    )
    command = build_benchmark_command(options)

    assert command[command.index("--model") + 1] == "custom-name"
    assert command[command.index("--result-dir") + 1] == "/tmp/result"
    assert command[-2:] == ["--percentile-metrics", "ttft"]


def test_tokenizer_resolution_prefers_explicit_and_environment(tmp_path) -> None:
    local_model = tmp_path / "local-model"
    local_model.mkdir()

    assert (
        resolve_tokenizer(
            "/models/explicit",
            environment={"HUST_VSPEC_TARGET_MODEL": "/models/environment"},
            candidates=(local_model,),
        )
        == "/models/explicit"
    )
    assert (
        resolve_tokenizer(
            environment={"HUST_VSPEC_TARGET_MODEL": "/models/environment"},
            candidates=(local_model,),
        )
        == "/models/environment"
    )


def test_tokenizer_resolution_uses_first_existing_candidate(tmp_path) -> None:
    missing = tmp_path / "missing"
    local_model = tmp_path / "local-model"
    local_model.mkdir()

    assert resolve_tokenizer(environment={}, candidates=(missing, local_model)) == str(local_model)
