# Draft B16 优化检查点（更新至 2026-09-23 UTC）

本页为历史证据导航。完整配置、测量、实现说明、复现命令、失败和负收益已原样保存在
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e01d8b6870dcbefd474b1102ef7f01e2a662ec9/archive/mod-targeted-20261023/vspec-b16-frontier/snapshot/docs/draft_b16_optimization_checkpoint_20260915.md)。旧文中的“当前”“未提交”和优化计划仅描述当时状态，
不代表现在的默认配置、运行状态或补测授权。迁移不新增性能或正确性验收结论。

## 可核验原始数据

- [benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200/GSM8K-n200.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200/GSM8K-n200.json)
- [benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200_r2/GSM8K-n200.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_strict_temp0_20260923/historical669_repro_n200_r2/GSM8K-n200.json)

## 引用可用性与边界

以下仅说明本次已固定的六份 vSpec capsule 清单的覆盖范围，不是全组织永久丢失判定。
没有用相邻运行、摘要或新实验补成原件。

- `/data/datasets/gsm8k/test.parquet` — 固定清单内未匹配，原件不可在本集合回验。
- `/tmp/vspec_draft_direct_rep_deep_profile_v3.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_recovery_20260914/` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_recovery_20260914/resume_n50` — 历史命令中的未来输出示例，不算缺失实测原件。
- `benchmark_results/draft_b16_resume_20260922/` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_resume_20260922/p5_split_margin4_n200_detailed/GSM8K-n200.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_resume_20260922/p5_split_margin4_n200_detailed_r2/GSM8K-n200.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_resume_20260922/p5_split_margin5p25_n200_detailed/GSM8K-n200.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_resume_20260922/p5_split_margin5p25_n200_detailed_r2/GSM8K-n200.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/gsm8k-baseline-20260901-vllm-hust/materialized.jsonl` — 固定清单内未匹配，原件不可在本集合回验。

保留 strict 与 relaxed acceptance、质量下降、筛选与全量、异步/同步对照差异；不能将旧工程目标当作新的公平配对结论。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确核查清单及未匹配路径见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
