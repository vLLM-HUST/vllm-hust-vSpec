#!/usr/bin/env bash

set -Eeuo pipefail

MODE=${1:-pair}
SPEC_GAMMA=${SPEC_GAMMA:-6}
ADAPTIVE_SPECULATION=${ADAPTIVE_SPECULATION:-1}
if [[ ! "${SPEC_GAMMA}" =~ ^(2|4|6)$ ]]; then
  echo "SPEC_GAMMA must be 2, 4, or 6" >&2
  exit 2
fi
if [[ "${ADAPTIVE_SPECULATION}" == 1 ]]; then
  if [[ "${SPEC_GAMMA}" != 6 ]]; then
    echo "adaptive MTP2 uses SPEC_GAMMA=6 as the 2/4/6 search upper bound" >&2
    exit 2
  fi
  SPEC_MODE=adaptive
elif [[ "${ADAPTIVE_SPECULATION}" == 0 ]]; then
  SPEC_MODE="mtp${SPEC_GAMMA}"
else
  echo "ADAPTIVE_SPECULATION must be 0 or 1" >&2
  exit 2
fi
if [[ "${MODE}" != "pair" && "${MODE}" != "baseline" && "${MODE}" != "${SPEC_MODE}" ]]; then
  echo "usage: ADAPTIVE_SPECULATION=${ADAPTIVE_SPECULATION} SPEC_GAMMA=${SPEC_GAMMA} $0 {pair|baseline|${SPEC_MODE}}" >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${VENV:-/opt/vllm-hust-cann91}
MIN_SPEEDUP=${MIN_SPEEDUP:-1.10}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}

if [[ "${MODE}" == "pair" ]]; then
  PAIR_DIR=${OUTPUT_DIR:-${ROOT}/benchmark_results/qwen35_frontier_mtp2_gsm8k/${RUN_ID}-pair}
  mkdir -p "${PAIR_DIR}"
  OUTPUT_DIR="${PAIR_DIR}/baseline" RUN_ID="${RUN_ID}" \
    SPEC_GAMMA="${SPEC_GAMMA}" ADAPTIVE_SPECULATION="${ADAPTIVE_SPECULATION}" \
    "$0" baseline
  OUTPUT_DIR="${PAIR_DIR}/${SPEC_MODE}" RUN_ID="${RUN_ID}" \
    SPEC_GAMMA="${SPEC_GAMMA}" ADAPTIVE_SPECULATION="${ADAPTIVE_SPECULATION}" \
    "$0" "${SPEC_MODE}"
  "${VENV}/bin/python" - \
    "${PAIR_DIR}/baseline/gsm8k.json" \
    "${PAIR_DIR}/${SPEC_MODE}/gsm8k.json" \
    "${PAIR_DIR}/comparison.json" \
    "${MIN_SPEEDUP}" "${SPEC_GAMMA}" "${ADAPTIVE_SPECULATION}" <<'PY'
import json
import sys
from pathlib import Path

baseline_path, mtp_path, output_path, minimum, gamma, adaptive = sys.argv[1:]
label = "adaptive_mtp2" if adaptive == "1" else f"mtp{gamma}"
baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
mtp = json.loads(Path(mtp_path).read_text(encoding="utf-8"))
if baseline["completed"] != mtp["completed"]:
    raise SystemExit(f"baseline and {label} completed request counts differ")
if baseline["total_input_tokens"] != mtp["total_input_tokens"]:
    raise SystemExit(f"baseline and {label} input token counts differ")
output_token_delta = abs(baseline["total_output_tokens"] - mtp["total_output_tokens"])
if output_token_delta / max(baseline["total_output_tokens"], mtp["total_output_tokens"]) > 0.001:
    raise SystemExit(f"baseline and {label} output token counts differ by more than 0.1%")
speedup = mtp["output_throughput"] / baseline["output_throughput"]
summary = {
    "baseline_output_throughput": baseline["output_throughput"],
    f"{label}_output_throughput": mtp["output_throughput"],
    "baseline_output_tokens": baseline["total_output_tokens"],
    f"{label}_output_tokens": mtp["total_output_tokens"],
    "gamma": int(gamma),
    "adaptive_speculation": adaptive == "1",
    "speedup": speedup,
    "minimum_speedup": float(minimum),
    "passed": speedup >= float(minimum),
}
Path(output_path).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not summary["passed"]:
    raise SystemExit(
        f"{label} speedup {speedup:.3f}x is below required {float(minimum):.3f}x"
    )
PY
  exit 0
fi

CANN_ENV=${CANN_ENV:-${VENV}/Ascend/cann-9.1.0/set_env.sh}
VLLM_SOURCE=${VLLM_SOURCE:-/root/data/vllm-hust-latest}
ASCEND_SOURCE=${ASCEND_SOURCE:-/root/data/vllm-ascend-hust-latest}
MODEL=${MODEL:-/workspace/models/Qwen3.5-35B-A3B}
DATASET=${DATASET:-/root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl}
NPU_IDS=${NPU_IDS:-0,1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18185}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-16}
MTP_LOCAL_ARGMAX=${MTP_LOCAL_ARGMAX:-0}
MTP_GRAPH_TRACE=${MTP_GRAPH_TRACE:-0}
MTP_COHORT_REFILL=${MTP_COHORT_REFILL:-1}
MTP_COHORT_REFILL_THRESHOLD=${MTP_COHORT_REFILL_THRESHOLD:-8}
ENABLE_PROFILE=${ENABLE_PROFILE:-0}
ASCEND_ADDITIONAL_CONFIG=${ASCEND_ADDITIONAL_CONFIG:-'{"enable_cpu_binding":true}'}
NUM_PROMPTS=${NUM_PROMPTS:-200}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/benchmark_results/qwen35_frontier_mtp2_gsm8k/${RUN_ID}-${MODE}}
SERVED_MODEL_NAME="qwen3.5-35b-a3b-frontier-${MODE}"
VLLM_BIN=${VLLM_BIN:-${VENV}/bin/vllm}
VSPEC_BIN=${VSPEC_BIN:-${VENV}/bin/vllm-hust-vspec}

for path in \
  "${CANN_ENV}" \
  "${VLLM_BIN}" \
  "${VSPEC_BIN}" \
  "${DATASET}" \
  "${MODEL}/model.safetensors.index.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "required path does not exist: ${path}" >&2
    exit 1
  fi
done
if compgen -G "${MODEL}/*.incomplete" >/dev/null; then
  echo "model download is incomplete: ${MODEL}" >&2
  exit 1
fi

# Match the CANN runtime to the vLLM-Ascend extension before Python imports it.
source "${CANN_ENV}"
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
export VLLM_PLUGINS=ascend,vspec
export HUST_VSPEC_MTP_COHORT_REFILL="${MTP_COHORT_REFILL}"
export HUST_VSPEC_MTP_COHORT_REFILL_THRESHOLD="${MTP_COHORT_REFILL_THRESHOLD}"
if [[ "${MTP_GRAPH_TRACE}" == 1 ]]; then
  export HUST_VSPEC_MTP_GRAPH_TRACE=1
fi

if [[ "${MODE}" == "baseline" ]]; then
  # Load vSpec's Qwen3.5 host compatibility in both arms. This arm remains
  # target-only because no speculative config is passed to vLLM.
  export HUST_VSPEC_ENABLED=1
  export HUST_VSPEC_METHOD=mtp
  export HUST_VSPEC_ADAPTIVE_SPECULATION=0
  export HUST_VSPEC_MTP_STRICT_GRAPH=1
  export HUST_VSPEC_MAX_NUM_SEQS="${MAX_NUM_SEQS}"
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
    --max-model-len 262144
    --gpu-memory-utilization 0.9
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens 8192
    --seed 0
    --scheduling-policy fcfs
    --distributed-executor-backend mp
    --disable-custom-all-reduce
    --no-trust-remote-code
    --load-format auto
    --generation-config vllm
    --no-enable-log-requests
    --uvicorn-log-level info
    --enable-prefix-caching
    --enable-chunked-prefill
    --language-model-only
    --async-scheduling
    --no-enforce-eager
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --cudagraph-capture-sizes 1 2 4 8 16
    --enable-expert-parallel
    --additional-config "${ASCEND_ADDITIONAL_CONFIG}"
  )
else
  case "${MTP_LOCAL_ARGMAX}" in
    0) MTP_ARGMAX_FLAG=--no-mtp-local-argmax-reduction ;;
    1) MTP_ARGMAX_FLAG=--mtp-local-argmax-reduction ;;
    *) echo "MTP_LOCAL_ARGMAX must be 0 or 1" >&2; exit 2 ;;
  esac
  if [[ "${ADAPTIVE_SPECULATION}" == 1 ]]; then
    FRONTIER_OPTIONS=(--protocol qwen35-frontier-mtp2)
  else
    FRONTIER_OPTIONS=(
      --config "${ROOT}/configs/qwen35-35b-a3b-frontier-mtp2.toml"
      --gamma "${SPEC_GAMMA}"
      --no-adaptive-speculation
    )
  fi
  SERVER_CMD=(
    "${VSPEC_BIN}"
    "${FRONTIER_OPTIONS[@]}"
    --target-model "${MODEL}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --device "${NPU_IDS}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    "${MTP_ARGMAX_FLAG}"
    --vllm-source "${VLLM_SOURCE}"
    --ascend-source "${ASCEND_SOURCE}"
    --vllm-executable "${VLLM_BIN}"
  )
fi

if [[ "${MODE}" != baseline && ( "${ENABLE_PROFILE}" == 1 || "${ASCEND_ADDITIONAL_CONFIG}" != '{"enable_cpu_binding":true}' ) ]]; then
  SERVER_CMD+=(-- --enable-expert-parallel --additional-config "${ASCEND_ADDITIONAL_CONFIG}")
fi

if [[ "${ENABLE_PROFILE}" == 1 ]]; then
  PROFILE_CONFIG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${OUTPUT_DIR}/profile\",\"delay_iterations\":50,\"max_iterations\":20,\"torch_profiler_with_stack\":false}"
  if [[ "${MODE}" != baseline ]]; then
    SERVER_CMD+=(--profiler-config "${PROFILE_CONFIG}")
  else
    SERVER_CMD+=(--profiler-config "${PROFILE_CONFIG}")
  fi
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
if [[ "${MAX_CONCURRENCY}" != "none" ]]; then
  if [[ ! "${MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_CONCURRENCY must be a positive integer or none" >&2
    exit 2
  fi
  BENCHMARK_COMMON+=(--max-concurrency "${MAX_CONCURRENCY}")
fi

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

print_command "${SERVER_CMD[@]}" >"${OUTPUT_DIR}/serve_cmd.txt"
print_command "${BENCHMARK_COMMON[@]}" >"${OUTPUT_DIR}/benchmark_common_cmd.txt"
printf 'MODE=%s\nSPEC_GAMMA=%s\nADAPTIVE_SPECULATION=%s\nMAX_NUM_SEQS=%s\nMAX_CONCURRENCY=%s\nMTP_LOCAL_ARGMAX=%s\nMTP_GRAPH_TRACE=%s\nMTP_COHORT_REFILL=%s\nMTP_COHORT_REFILL_THRESHOLD=%s\nENABLE_PROFILE=%s\nASCEND_ADDITIONAL_CONFIG=%s\n' \
  "${MODE}" "${SPEC_GAMMA}" "${ADAPTIVE_SPECULATION}" "${MAX_NUM_SEQS}" "${MAX_CONCURRENCY}" "${MTP_LOCAL_ARGMAX}" \
  "${MTP_GRAPH_TRACE}" "${MTP_COHORT_REFILL}" "${MTP_COHORT_REFILL_THRESHOLD}" \
  "${ENABLE_PROFILE}" "${ASCEND_ADDITIONAL_CONFIG}" >"${OUTPUT_DIR}/benchmark_env.txt"

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

BENCHMARK_RUN=(
  "${BENCHMARK_COMMON[@]}"
  --custom-output-len "${OUTPUT_TOKENS}"
  --num-prompts "${NUM_PROMPTS}"
  --save-result
  --result-dir "${OUTPUT_DIR}"
  --result-filename gsm8k.json
)
if [[ "${ENABLE_PROFILE}" == 1 ]]; then
  BENCHMARK_RUN+=(--profile)
fi
"${BENCHMARK_RUN[@]}" 2>&1 | tee "${OUTPUT_DIR}/benchmark.log"
if [[ "${ENABLE_PROFILE}" == 1 ]]; then
  if ! grep -Fq "Profiler started." "${OUTPUT_DIR}/server.log"; then
    echo "profiler did not start" >&2
    exit 1
  fi
  if ! rg --files "${OUTPUT_DIR}/profile" 2>/dev/null | rg -q 'profiler_info_0\.json$'; then
    echo "profiler raw data was not created" >&2
    exit 1
  fi
fi

for evidence in \
  "Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)" \
  "Capturing CUDA graphs (decode, FULL)" \
  "Graph capturing finished"; do
  if ! grep -Fq "${evidence}" "${OUTPUT_DIR}/server.log"; then
    echo "missing target graph evidence: ${evidence}" >&2
    exit 1
  fi
done
if [[ "${MODE}" != baseline ]] && ! grep -Fq \
  "Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL" \
  "${OUTPUT_DIR}/server.log"; then
  echo "MTP proposer did not enter FULL graph mode" >&2
  exit 1
fi
if grep -Fq "vSpec MTP refused an eager decode fallback" "${OUTPUT_DIR}/server.log"; then
  echo "strict MTP graph guard detected an eager fallback" >&2
  exit 1
fi
printf '%s graph validation passed.\n' "${MODE}"
