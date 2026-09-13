# ARC-Easy 同口径回归命令

本协议沿用工程基线文档中的容器、模型挂载、数据挂载和
`/run_dir/materialized.jsonl` 生成步骤。只替换“启动 vLLM”和“执行压测”两部分。

## Draft

启动服务：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method draft \
  --draft-model /model/Qwen2.5-0.5B-Instruct
```

执行压测：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec-bench --method draft
```

压测入口会按“显式 `--tokenizer`、`HUST_VSPEC_TARGET_MODEL`、容器
`/model`、宿主机 `/data/shared-models`”的顺序解析 tokenizer，因此容器外执行时
不需要手工覆盖默认路径。

## EAGLE

启动服务：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method eagle \
  --draft-model /model/Eagle-Qwen2.5-14B-Instruct
```

执行压测：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec-bench --method eagle
```

## 固定口径

- Target：`/model/Qwen2.5-14B-Instruct`
- dtype：FP16
- batch 上限：16
- `max_num_batched_tokens`：8192
- `max_model_len`：32768
- 端口：18180
- prefix caching：关闭
- chunked prefill：开启
- Graph：`FULL_DECODE_ONLY`
- 动态 gamma：默认开启，在线范围 `1..4`
- 数据：`/run_dir/materialized.jsonl`
- prompt 数：200
- 每条输出：256 token
- 请求速率：无限
- temperature：0
- seed：0

2026-09-13 的同卡结果、GSM8K B128 `1.518x` 恢复验证以及 B16/ARC-Easy
对照见 [`gsm8k_b128_recovery_and_arc_easy_20260913.md`](gsm8k_b128_recovery_and_arc_easy_20260913.md)。

服务端为了覆盖动态 gamma 的 Draft 和 Target 验证形状，会自动扩展 Graph capture
bucket。这是投机执行所需的图形状，其他公共服务参数与 Target-only 基线保持一致。

查看实际展开命令而不启动服务或发起请求：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method draft \
  --draft-model /model/Qwen2.5-0.5B-Instruct \
  --dry-run

/usr/local/python3.11.14/bin/vllm-hust-vspec-bench \
  --method draft \
  --dry-run
```
