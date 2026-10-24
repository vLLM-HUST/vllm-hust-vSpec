# Qwen3.5 Frontier 固定 MTP2 历史验证

本页为历史证据导航。完整配置、测量、实现说明、复现命令、失败和负收益已原样保存在
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e01d8b6870dcbefd474b1102ef7f01e2a662ec9/archive/mod-targeted-20261023/vspec-b16-frontier/snapshot/docs/qwen35_frontier_mtp2.md)。旧文中的“当前”“未提交”和优化计划仅描述当时状态，
不代表现在的默认配置、运行状态或补测授权。迁移不新增性能或正确性验收结论。

## 可核验原始数据

- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/aiperf/profile_export_aiperf.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/aiperf/profile_export_aiperf.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/run.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/run.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/target.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/target.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/aiperf/profile_export_aiperf.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/aiperf/profile_export_aiperf.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/run.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/run.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/target.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/target.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-baseline-final/gsm8k.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-baseline-final/gsm8k.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-mtp2-final/gsm8k.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/20260927-c4-mtp2-final/gsm8k.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c16.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c16.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c4.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c4.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c8.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-baseline/c8.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c16.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c16.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c4.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c4.json)
- [benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c8.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_frontier_mtp2_gsm8k/scaling-mtp2/c8.json)
- [benchmark_results/qwen35_long_context/20260930-mtp2-26case/results.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_long_context/20260930-mtp2-26case/results.json)

## 引用可用性与边界

以下仅说明本次已固定的六份 vSpec capsule 清单的覆盖范围，不是全组织永久丢失判定。
没有用相邻运行、摘要或新实验补成原件。

- `/tmp/vspec_qwen35_frontier_mtp2_stable.log` — 固定清单内未匹配，原件不可在本集合回验。

## 256K 正确性

原生长上下文正确性与 synthetic-sampler AgentX 性能分开，低并发收益不外推到全部并发。当前部署入口见 [B16 命令](qwen35_mtp2_b16_graph_benchmark_commands.md)；历史固定协议与命令保留在原文。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确核查清单及未匹配路径见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
