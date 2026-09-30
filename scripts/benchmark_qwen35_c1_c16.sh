#!/usr/bin/env bash

set -Eeuo pipefail

VENV=${VENV:-/opt/vllm-hust-cann91}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18185}
MODEL=${MODEL:-qwen3.5-35b-a3b-frontier-baseline}
TOKENIZER=${TOKENIZER:-/workspace/models/Qwen3.5-35B-A3B}
DATASET=${DATASET:-/root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR must be set}
NUM_PROMPTS=${NUM_PROMPTS:-64}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
CONCURRENCIES=${CONCURRENCIES:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16}

mkdir -p "${OUTPUT_DIR}"

if ! curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "service is not healthy at http://${HOST}:${PORT}" >&2
  exit 1
fi

COMMON=(
  "${VENV}/bin/vllm" bench serve
  --backend openai-chat
  --base-url "http://${HOST}:${PORT}"
  --endpoint /v1/chat/completions
  --model "${MODEL}"
  --tokenizer "${TOKENIZER}"
  --dataset-name custom
  --dataset-path "${DATASET}"
  --disable-shuffle
  --request-rate inf
  --temperature 0
  --seed 0
)

"${COMMON[@]}" \
  --max-concurrency 1 \
  --custom-output-len 32 \
  --num-prompts 2 \
  >"${OUTPUT_DIR}/warmup.log" 2>&1

for concurrency in ${CONCURRENCIES}; do
  if [[ ! "${concurrency}" =~ ^([1-9]|1[0-6])$ ]]; then
    echo "invalid concurrency ${concurrency}; expected 1..16" >&2
    exit 2
  fi
  echo "Running C${concurrency}"
  "${COMMON[@]}" \
    --max-concurrency "${concurrency}" \
    --custom-output-len "${OUTPUT_TOKENS}" \
    --num-prompts "${NUM_PROMPTS}" \
    --save-result \
    --result-dir "${OUTPUT_DIR}" \
    --result-filename "c${concurrency}.json" \
    2>&1 | tee "${OUTPUT_DIR}/c${concurrency}.log"
done

"${VENV}/bin/python" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("c*.json"), key=lambda item: int(item.stem[1:])):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "concurrency": int(path.stem[1:]),
        "completed": data["completed"],
        "failed": data["failed"],
        "output_throughput": data["output_throughput"],
        "mean_ttft_ms": data["mean_ttft_ms"],
        "mean_tpot_ms": data["mean_tpot_ms"],
        "mean_itl_ms": data["mean_itl_ms"],
    })
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
PY
