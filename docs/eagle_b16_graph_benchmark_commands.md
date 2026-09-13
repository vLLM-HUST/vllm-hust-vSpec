# EAGLE B16 Graph

## GSM8K - Terminal 1

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7

/opt/vllm-hust-cann91/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method eagle \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-model /data/shared-models/Eagle-Qwen2.5-14B-Instruct
```

## GSM8K - Terminal 2

```bash
/opt/vllm-hust-cann91/bin/vllm-hust-vspec-bench \
  --method eagle \
  --base-url http://127.0.0.1:18180 \
  --tokenizer /data/shared-models/Qwen2.5-14B-Instruct \
  --dataset-path /root/data/vllm-ascend-hust/benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl \
  --output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --temperature 0 \
  --seed 0 \
  --result-dir /root/data/vllm-hust-vSpec/benchmark_results/GSM8K-eagle-vspec \
  --result-filename GSM8K.json
```

## ARC-Easy - Terminal 1

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7

/opt/vllm-hust-cann91/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method eagle \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-model /data/shared-models/Eagle-Qwen2.5-14B-Instruct
```

## ARC-Easy - Terminal 2

```bash
/opt/vllm-hust-cann91/bin/vllm-hust-vspec-bench \
  --method eagle \
  --base-url http://127.0.0.1:18180 \
  --tokenizer /data/shared-models/Qwen2.5-14B-Instruct \
  --dataset-path /root/data/vllm-hust-vSpec/benchmark_results/arc_easy_reference/materialized.jsonl \
  --output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --temperature 0 \
  --seed 0 \
  --result-dir /root/data/vllm-hust-vSpec/benchmark_results/ARC-Easy-eagle-vspec \
  --result-filename ARC-Easy.json
```
