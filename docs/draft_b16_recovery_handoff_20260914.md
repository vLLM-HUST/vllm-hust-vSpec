# Draft B16 performance recovery handoff (2026-09-14)

本页为历史证据导航。完整配置、测量、实现说明、复现命令、失败和负收益已原样保存在
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e01d8b6870dcbefd474b1102ef7f01e2a662ec9/archive/mod-targeted-20261023/vspec-b16-frontier/snapshot/docs/draft_b16_recovery_handoff_20260914.md)。旧文中的“当前”“未提交”和优化计划仅描述当时状态，
不代表现在的默认配置、运行状态或补测授权。迁移不新增性能或正确性验收结论。

## 可核验原始数据

本篇列出的旧结果目录尚未在下述固定清单内匹配；历史正文不等于原件恢复。

## 引用可用性与边界

以下仅说明本次已固定的六份 vSpec capsule 清单的覆盖范围，不是全组织永久丢失判定。
没有用相邻运行、摘要或新实验补成原件。

- `/tmp/vspec_draft_clean_g2_p4_profile.json` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/b16_draft_recovery_20260913` — 固定清单内未匹配，原件不可在本集合回验。
- `benchmark_results/draft_b16_recovery_20260914/` — 固定清单内未匹配，原件不可在本集合回验。

保留 strict 与 relaxed acceptance、质量下降、筛选与全量、异步/同步对照差异；不能将旧工程目标当作新的公平配对结论。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确核查清单及未匹配路径见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
