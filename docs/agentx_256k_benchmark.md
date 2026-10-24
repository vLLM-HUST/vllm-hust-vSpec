# AgentX 256K 评测接入

本页为历史证据导航。[完整原文与复现方法](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/51604897aabb9a856b311fa58fc6749938d0378e/archive/mod-targeted-20261023/vspec-agentx-school/snapshot/docs/agentx_256k_benchmark.md)已原样归档，保留全部配置、数值、失败与适用边界，不代表当前默认部署或新验收结论。

## 可核验原始数据

- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/aiperf/profile_export_aiperf.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/aiperf/profile_export_aiperf.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/run.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/run.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/target.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/target.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/aiperf/profile_export_aiperf.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/aiperf/profile_export_aiperf.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/run.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/run.json)
- [benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/target.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/target.json)

## 原件覆盖边界

以下成员未在本次固定的七份 vSpec capsule 清单内匹配；不作全组织永久丢失判定，不以摘要冒充逐请求原件，也不启动补测。

- `benchmark_results/qwen35_agentx_256k/20260930-current-pair/target-only/harness.log`
- `benchmark_results/qwen35_agentx_256k/20260930-current-pair/mtp2/harness.log`

Synthetic sampler 的 AgentX serving 性能不证明生成内容正确性；闭环两臂推进的请求组合可能不同。自然 sampler 的[长上下文正确性](qwen35_frontier_mtp2.md#256k-正确性)单独读取。配置模板仍在 `configs/agentx-qwen35-frontier-*.example.json`，SPEED-Bench AL 来源见[对应原件入口](qwen35_speedbench_coding_al.md)，不能改用 GSM8K AL。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确绑定和缺失范围见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
