#!/usr/bin/env bash

set -Eeuo pipefail

MODE=${1:-}
if [[ "${MODE}" == "baseline" ]]; then
  GAMMA=0
elif [[ "${MODE}" =~ ^mtp([1-6])$ ]]; then
  GAMMA=${BASH_REMATCH[1]}
else
  echo "usage: $0 {baseline|mtp1|mtp2|mtp3|mtp4|mtp5|mtp6}" >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CANN_ENV=${CANN_ENV:-/opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh}
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script does not exist: ${CANN_ENV}" >&2
  exit 1
fi
# The selected vLLM-Ascend extension is built against CANN 9.1. Source its
# matching runtime before Python starts so the custom-op loader cannot bind 9.0.
source "${CANN_ENV}"

VENV=${VENV:-/opt/vllm-hust-cann91}
VLLM_SOURCE=${VLLM_SOURCE:-/root/data/vllm-hust-latest}
ASCEND_SOURCE=${ASCEND_SOURCE:-/root/data/vllm-ascend-hust-latest}
MODEL=${MODEL:-/workspace/models/Qwen3.5-35B-A3B}
DATASET=${DATASET:-/root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl}
CONFIG=${CONFIG:-${ROOT}/configs/qwen35-35b-a3b-mtp2.toml}
NPU_IDS=${NPU_IDS:-0,1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18185}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_PROMPTS=${NUM_PROMPTS:-200}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
ENABLE_REDUCE_SAMPLE=${ENABLE_REDUCE_SAMPLE:-0}
ENABLE_LOCAL_ARGMAX=${ENABLE_LOCAL_ARGMAX:-0}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/benchmark_results/qwen35_mtp2_gsm8k/${RUN_ID}-${MODE}}
SERVED_MODEL_NAME="qwen3.5-35b-a3b-${MODE}"

case "${ENABLE_REDUCE_SAMPLE}" in
  0)
    REDUCE_SAMPLE_JSON=false
    ;;
  1)
    REDUCE_SAMPLE_JSON=true
    ;;
  *)
    echo "ENABLE_REDUCE_SAMPLE must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${ENABLE_LOCAL_ARGMAX}" in
  0)
    LOCAL_ARGMAX_FLAG=--no-mtp-local-argmax-reduction
    ;;
  1)
    LOCAL_ARGMAX_FLAG=--mtp-local-argmax-reduction
    ;;
  *)
    echo "ENABLE_LOCAL_ARGMAX must be 0 or 1" >&2
    exit 2
    ;;
esac
ASCEND_ADDITIONAL_CONFIG=$(printf \
  '{"enable_cpu_binding":true,"multistream_overlap_shared_expert":true,"enable_reduce_sample":%s}' \
  "${REDUCE_SAMPLE_JSON}")

VLLM_BIN=${VLLM_BIN:-${VENV}/bin/vllm}
VSPEC_BIN=${VSPEC_BIN:-${VENV}/bin/vllm-hust-vspec}

for path in "${VLLM_BIN}" "${VSPEC_BIN}" "${DATASET}" "${CONFIG}" "${MODEL}/model.safetensors.index.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path does not exist: ${path}" >&2
    exit 1
  fi
done
if compgen -G "${MODEL}/*.incomplete" >/dev/null; then
  echo "model download is incomplete: ${MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}/src:${VLLM_SOURCE}:${ASCEND_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_AOT_COMPILE=0
export VLLM_NO_USAGE_STATS=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

if [[ "${MODE}" == "baseline" ]]; then
  # Load the same Qwen3.5 host-compatibility and strict graph guards in both
  # arms. The baseline remains target-only because it has no speculative config.
  export VLLM_PLUGINS=ascend,vspec
  export HUST_VSPEC_ENABLED=1
  export HUST_VSPEC_METHOD=mtp
  export HUST_VSPEC_ADAPTIVE_SPECULATION=0
  export HUST_VSPEC_MTP_STRICT_GRAPH=1
  export HUST_VSPEC_MAX_NUM_SEQS="${BATCH_SIZE}"
  SERVER_CMD=(
    "${VLLM_BIN}" serve "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --dtype bfloat16
    --kv-cache-dtype auto
    --block-size 128
    --tensor-parallel-size 2
    --pipeline-parallel-size 1
    --data-parallel-size 1
    --max-model-len 4096
    --gpu-memory-utilization 0.9
    --max-num-seqs "${BATCH_SIZE}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --seed 0
    --scheduling-policy fcfs
    --distributed-executor-backend mp
    --disable-custom-all-reduce
    --no-trust-remote-code
    --load-format auto
    --generation-config vllm
    --no-enable-log-requests
    --uvicorn-log-level info
    --no-enable-prefix-caching
    --enable-chunked-prefill
    --language-model-only
    --no-async-scheduling
    --no-enforce-eager
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --cudagraph-capture-sizes 1 2 4 8 16
    --enable-expert-parallel
    --additional-config "${ASCEND_ADDITIONAL_CONFIG}"
  )
else
  export VLLM_PLUGINS=ascend,vspec
  SERVER_CMD=(
    "${VSPEC_BIN}"
    --config "${CONFIG}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${NPU_IDS}"
    --gamma "${GAMMA}"
    "${LOCAL_ARGMAX_FLAG}"
    --max-num-seqs "${BATCH_SIZE}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --
    --enable-expert-parallel
    --additional-config "${ASCEND_ADDITIONAL_CONFIG}"
  )
fi

BENCHMARK_COMMON=(
  "${VLLM_BIN}" bench serve
  --backend openai-chat
  --base-url "http://${HOST}:${PORT}"
  --endpoint /v1/chat/completions
  --model "${SERVED_MODEL_NAME}"
  --tokenizer "${MODEL}"
  --dataset-name custom
  --dataset-path "${DATASET}"
  --disable-shuffle
  --request-rate inf
  --temperature 0
  --seed 0
)

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

print_command "${SERVER_CMD[@]}" >"${OUTPUT_DIR}/serve_cmd.txt"
print_command "${BENCHMARK_COMMON[@]}" >"${OUTPUT_DIR}/benchmark_common_cmd.txt"
printf 'ENABLE_REDUCE_SAMPLE=%s\n' "${ENABLE_REDUCE_SAMPLE}" \
  >"${OUTPUT_DIR}/benchmark_env.txt"
printf 'ENABLE_LOCAL_ARGMAX=%s\n' "${ENABLE_LOCAL_ARGMAX}" \
  >>"${OUTPUT_DIR}/benchmark_env.txt"

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
until curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "server exited before becoming healthy" >&2
    tail -n 200 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "server startup exceeded ${STARTUP_TIMEOUT}s" >&2
    tail -n 200 "${OUTPUT_DIR}/server.log" >&2
    exit 1
  fi
  sleep 2
done

"${BENCHMARK_COMMON[@]}" \
  --custom-output-len 32 \
  --num-prompts 8 \
  >"${OUTPUT_DIR}/warmup.log" 2>&1

"${BENCHMARK_COMMON[@]}" \
  --custom-output-len "${OUTPUT_TOKENS}" \
  --num-prompts "${NUM_PROMPTS}" \
  --save-result \
  --result-dir "${OUTPUT_DIR}" \
  --result-filename gsm8k.json \
  2>&1 | tee "${OUTPUT_DIR}/benchmark.log"

if [[ "${MODE}" != "baseline" ]]; then
  if ! grep -Fq \
    "Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL" \
    "${OUTPUT_DIR}/server.log"; then
    echo "MTP proposer did not enter FULL graph mode" >&2
    exit 1
  fi
  if ! grep -Fq "Graph capturing finished" "${OUTPUT_DIR}/server.log"; then
    echo "target graph capture did not finish" >&2
    exit 1
  fi
  if grep -Fq "vSpec MTP refused an eager decode fallback" \
    "${OUTPUT_DIR}/server.log"; then
    echo "strict MTP graph guard detected an eager fallback" >&2
    exit 1
  fi
  printf '%s graph validation passed: target and proposer used graphs.\n' "${MODE}"
fi
