# Qwen3.5-35B MTP2 GSM8K B16 Graph 测试命令

## 模型准备

只需要一个模型目录；MTP head 已包含在 target checkpoint 中，不需要单独的 Drafter：

| 用途 | 模型 | 容器内路径 |
|---|---|---|
| Target + MTP2 head | `Qwen/Qwen3.5-35B-A3B` | `/model/Qwen3.5-35B-A3B` |

vSpec 安装器不会下载模型。模型不存在时，由用户在联网环境手动执行：

```bash
hf download Qwen/Qwen3.5-35B-A3B \
  --local-dir /data/shared_models/Qwen3.5-35B-A3B
```

启动容器时将 `/data/shared_models` 只读挂载到 `/model`。GSM8K 需提前物化为
`/run_dir/materialized.jsonl`，每行格式为 `{"prompt":"..."}`。

## 启动 vLLM

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1=1
export VLLM_PLUGINS=ascend,vspec
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_AOT_COMPILE=0
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1

/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol qwen35-frontier-mtp2 \
  --target-model /model/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b-mtp2-vspec \
  --host 127.0.0.1 \
  --port 18180 \
  --device 0,1 \
  --max-num-seqs 16
```

该协议默认开启自动 gamma 和自动 refill。MTP2 head 的在线 gamma 候选为 `2/4/6`，
`6` 是搜索上限而不是固定投机长度；无需额外传入 `--gamma`、
`--adaptive-speculation` 或 refill 环境变量。

## 执行压测

```bash
/usr/local/python3.11.14/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18180 \
  --endpoint /v1/chat/completions \
  --model qwen3.5-35b-a3b-mtp2-vspec \
  --tokenizer /model/Qwen3.5-35B-A3B \
  --dataset-name custom \
  --dataset-path /run_dir/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --max-concurrency 16 \
  --temperature 0 \
  --seed 0 \
  --save-result \
  --result-dir /run_dir/benchmark_results/Qwen3.5-MTP2-GSM8K-B16 \
  --result-filename GSM8K.json
```
