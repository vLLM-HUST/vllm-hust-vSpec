# Qwen3.5 Frontier MTP2 GSM8K 测试命令

## 需要的模型

本测试只需要一个 checkpoint：

| 用途 | 模型 | 容器内示例路径 |
|---|---|---|
| Target + 原生 MTP head | `Qwen/Qwen3.5-35B-A3B` | `/model/Qwen3.5-35B-A3B` |

Qwen3.5-35B-A3B checkpoint 已内置 MTP head，不需要额外下载 Draft、EAGLE 或 EAGLE3
模型。模型目录至少应包含 `config.json`、tokenizer 文件、
`model.safetensors.index.json` 和完整的 safetensors 分片，且不能残留 `*.incomplete`。

模型不存在时可在联网环境下载：

```bash
hf download Qwen/Qwen3.5-35B-A3B \
  --local-dir /data/shared_models/Qwen3.5-35B-A3B
```

启动容器时将模型目录只读挂载为 `/model`，并向容器暴露两张 Ascend 910B2：

```bash
--device /dev/davinci0 \
--device /dev/davinci1 \
-v /data/shared_models:/model:ro
```

## 检查模型与数据集

```bash
ls -lh /model/Qwen3.5-35B-A3B/config.json
ls -lh /model/Qwen3.5-35B-A3B/model.safetensors.index.json
find /model/Qwen3.5-35B-A3B -maxdepth 1 -name '*.incomplete' -print
ls -lh /data/assets/gsm8k
```

最后一条 `find` 命令必须没有输出。如果 GSM8K 挂载目录内是指向容器外路径的符号链接，
应改为直接挂载符号链接指向的真实目录。

## 生成 materialized.jsonl

GSM8K custom loader 使用每行一个 `{"prompt": "..."}` 的 JSONL 文件。本次已验证口径
直接使用原始 `question`，不额外拼接提示词。

原始数据为 JSONL 时：

```bash
python3 - <<'PY'
import json

raw_path = "/data/assets/gsm8k/test.jsonl"
out_path = "/run_dir/materialized.jsonl"
with open(raw_path, encoding="utf-8") as source, open(
    out_path, "w", encoding="utf-8"
) as output:
    count = 0
    for line in source:
        if not line.strip():
            continue
        example = json.loads(line)
        output.write(json.dumps({"prompt": example["question"]}, ensure_ascii=False) + "\n")
        count += 1
print(f"done, rows={count}")
PY
```

原始数据为 Parquet 时：

```bash
python3 - <<'PY'
import json
import pandas as pd

raw_path = "/data/assets/gsm8k/test-00000-of-00001.parquet"
out_path = "/run_dir/materialized.jsonl"
frame = pd.read_parquet(raw_path)
with open(out_path, "w", encoding="utf-8") as output:
    for question in frame["question"]:
        output.write(json.dumps({"prompt": question}, ensure_ascii=False) + "\n")
print(f"done, rows={len(frame)}")
PY
```

## 启动 vLLM MTP2

以下命令使用 vSpec `0.14.2` 的固定 Frontier 协议。服务端容量是 16，客户端正式压测
并发是 4；两者不是同一个参数。

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

vllm-hust-vspec \
  --protocol qwen35-frontier-mtp2 \
  --target-model /model/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b-frontier-mtp2 \
  --host 127.0.0.1 \
  --port 18185 \
  --device 0,1 \
  --max-num-seqs 16 \
  --no-mtp-local-argmax-reduction
```

该协议固定以下关键参数：

| 参数 | 值 |
|---|---|
| 方法 | 原生 MTP |
| 投机 token 数 | 2 |
| Tensor / Expert Parallel | TP2 / EP2 |
| dtype | BF16 |
| 最大上下文 | 262,144 |
| Prefix cache | 开启 |
| Chunked prefill | 开启 |
| Async scheduling | 开启 |
| Target graph | `FULL_AND_PIECEWISE` |
| MTP proposer graph | `FULL` |
| 最大序列数 | 16 |
| 最大 batch token | 8,192 |

服务启动日志必须同时出现以下内容：

```text
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
Capturing CUDA graphs (decode, FULL)
Wrapping draft model with ACLGraphWrapper: runtime_mode=FULL
Application startup complete
```

## 预热

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18185 \
  --endpoint /v1/chat/completions \
  --model qwen3.5-35b-a3b-frontier-mtp2 \
  --tokenizer /model/Qwen3.5-35B-A3B \
  --dataset-name custom \
  --dataset-path /run_dir/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 32 \
  --num-prompts 8 \
  --request-rate inf \
  --max-concurrency 4 \
  --temperature 0 \
  --seed 0
```

## 执行正式压测

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18185 \
  --endpoint /v1/chat/completions \
  --model qwen3.5-35b-a3b-frontier-mtp2 \
  --tokenizer /model/Qwen3.5-35B-A3B \
  --dataset-name custom \
  --dataset-path /run_dir/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --max-concurrency 4 \
  --temperature 0 \
  --seed 0 \
  --save-result \
  --result-dir /run_dir/benchmark_results/Qwen3.5-MTP2-GSM8K \
  --result-filename GSM8K.json
```

正式口径为 GSM8K 固定顺序前 200 条、每条最多生成 256 token、temperature 0、seed 0、
客户端并发 4。已验证结果和 target-only 公平对照见
[`qwen35_frontier_mtp2.md`](qwen35_frontier_mtp2.md)。
