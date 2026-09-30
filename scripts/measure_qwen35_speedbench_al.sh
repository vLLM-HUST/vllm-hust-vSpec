#!/usr/bin/env bash

set -Eeuo pipefail

THINKING_MODE=${1:-on}
if [[ "${THINKING_MODE}" != "on" && "${THINKING_MODE}" != "off" ]]; then
  echo "usage: $0 {on|off}" >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-/opt/vllm-hust-cann91}
CANN_ENV=${CANN_ENV:-${VENV}/Ascend/cann-9.1.0/set_env.sh}
MODEL=${MODEL:-/workspace/models/Qwen3.5-35B-A3B}
DATASET_DIR=${DATASET_DIR:-/tmp/vspec-speedbench-data}
NPU_IDS=${NPU_IDS:-0,1}
PORT=${PORT:-18185}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/benchmark_results/qwen35_speedbench_al/$(date -u +%Y%m%d-%H%M%S)-${THINKING_MODE}}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
DATASET_TIMEOUT=${DATASET_TIMEOUT:-1800}
VLLM_BIN=${VLLM_BIN:-${VENV}/bin/vllm}
VSPEC_BIN=${VSPEC_BIN:-${VENV}/bin/vllm-hust-vspec}
SERVED_MODEL_NAME=qwen3.5-35b-a3b-speedbench-mtp2

for path in "${CANN_ENV}" "${MODEL}/model.safetensors.index.json" \
  "${VLLM_BIN}" "${VSPEC_BIN}"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path does not exist: ${path}" >&2
    exit 1
  fi
done
if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "port ${PORT} already serves a healthy endpoint" >&2
  exit 1
fi

source "${CANN_ENV}"
mkdir -p "${OUTPUT_DIR}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:/root/data/vllm-hust-latest:/root/data/vllm-ascend-hust-latest${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_AOT_COMPILE=0
export VLLM_NO_USAGE_STATS=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_PLUGINS=ascend,vspec

if [[ "${THINKING_MODE}" == "on" ]]; then
  TEMPERATURE=0.6
  TOP_P=0.95
  TOP_K=20
  PRESENCE_PENALTY=0.0
  CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
else
  TEMPERATURE=0.7
  TOP_P=0.8
  TOP_K=20
  PRESENCE_PENALTY=1.5
  CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi

SERVER_CMD=(
  "${VSPEC_BIN}" --protocol qwen35-frontier-mtp2
  --target-model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 127.0.0.1 --port "${PORT}"
  --device "${NPU_IDS}"
  --vllm-source /root/data/vllm-hust-latest
  --ascend-source /root/data/vllm-ascend-hust-latest
  --vllm-executable "${VLLM_BIN}"
)
BENCH_CMD=(
  "${VLLM_BIN}" bench serve
  --model "${SERVED_MODEL_NAME}"
  --tokenizer "${MODEL}"
  --host 127.0.0.1 --port "${PORT}"
  --dataset-name speed_bench
  --dataset-path "${DATASET_DIR}"
  --speed-bench-category coding
  --speed-bench-output-len 4096
  --num-prompts -1
  --max-concurrency 1
  --request-rate inf
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --presence-penalty "${PRESENCE_PENALTY}"
  --chat-template-kwargs "${CHAT_TEMPLATE_KWARGS}"
  --seed 0
  --save-result --save-detailed
  --result-dir "${OUTPUT_DIR}"
  --result-filename result.json
)

printf '%q ' "${SERVER_CMD[@]}" >"${OUTPUT_DIR}/serve_cmd.txt"
printf '\n' >>"${OUTPUT_DIR}/serve_cmd.txt"
printf '%q ' "${BENCH_CMD[@]}" >"${OUTPUT_DIR}/bench_cmd.txt"
printf '\n' >>"${OUTPUT_DIR}/bench_cmd.txt"
SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -- "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
setsid "${SERVER_CMD[@]}" >"${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 100 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "server startup timed out" >&2
    tail -n 100 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  sleep 2
done

deadline=$((SECONDS + DATASET_TIMEOUT))
until [[ -s "${DATASET_DIR}/qualitative.jsonl" ]]; do
  if ((SECONDS >= deadline)); then
    echo "dataset preparation timed out" >&2
    exit 1
  fi
  sleep 2
done
sha256sum "${DATASET_DIR}/qualitative.jsonl" >"${OUTPUT_DIR}/dataset.sha256"

"${BENCH_CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/bench.log"

"${VENV}/bin/python" - "${OUTPUT_DIR}/result.json" \
  "${OUTPUT_DIR}/al_summary.json" "${THINKING_MODE}" <<'PY'
import json
import sys
from pathlib import Path

result_path, summary_path, thinking_mode = map(Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
drafts = result.get("spec_decode_num_drafts", 0)
accepted = result.get("spec_decode_accepted_tokens", 0)
if result.get("failed", 0) or not result.get("completed", 0) or drafts <= 0:
    raise SystemExit("benchmark incomplete or speculative counters missing")
calculated_al = 1 + accepted / drafts
reported_al = result.get("spec_decode_acceptance_length")
if reported_al is None or abs(calculated_al - reported_al) > 1e-6:
    raise SystemExit("benchmark acceptance length does not match raw counters")
summary = {
    "model": "Qwen3.5-35B-A3B",
    "method": "mtp",
    "num_speculative_tokens": 2,
    "thinking_mode": str(thinking_mode),
    "dataset": "SPEED-Bench qualitative coding",
    "output_len": 4096,
    "concurrency": 1,
    "completed": result["completed"],
    "failed": result.get("failed", 0),
    "num_drafts": drafts,
    "num_accepted_tokens": accepted,
    "acceptance_length": calculated_al,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

if ! grep -Fq "Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL" \
  "${OUTPUT_DIR}/server.log"; then
  echo "MTP proposer graph evidence missing" >&2
  exit 1
fi
if grep -Fq "vSpec MTP refused an eager decode fallback" \
  "${OUTPUT_DIR}/server.log"; then
  echo "strict graph guard detected an eager fallback" >&2
  exit 1
fi
