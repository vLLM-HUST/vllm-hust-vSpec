# EAGLE 动态 gamma 0～4 修复与实测

本页为历史证据导航。完整配置、测量、实现说明、复现命令、失败和负收益已原样保存在
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e01d8b6870dcbefd474b1102ef7f01e2a662ec9/archive/mod-targeted-20261023/vspec-b16-frontier/snapshot/docs/eagle_dynamic_gamma_0to4_gsm8k_b128_report.md)。旧文中的“当前”“未提交”和优化计划仅描述当时状态，
不代表现在的默认配置、运行状态或补测授权。迁移不新增性能或正确性验收结论。

## 可核验原始数据

- [benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/eagle_online_g0to4_all_arms_margin4_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_all_arms_margin4_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_g0to4_all_arms_margin4_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_all_arms_margin4_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/eagle_online_g0to4_all_arms_strict_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_all_arms_strict_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_g0to4_all_arms_strict_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_all_arms_strict_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/eagle_online_g0to4_safe_burnin_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_safe_burnin_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_g0to4_safe_burnin_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_g0to4_safe_burnin_full_b128_gsm8k_n200_max512.log)

## 引用可用性与边界

以下仅说明本次已固定的六份 vSpec capsule 清单的覆盖范围，不是全组织永久丢失判定。
没有用相邻运行、摘要或新实验补成原件。

- `benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.log` — 固定清单内未匹配，原件不可在本集合回验。

控制器候选范围不等于该轮实际遍历全部候选；吞吐准入不等于 TTFT/SLO goodput 通过。Graph 与 eager、质量约束仍按原文分别读取。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确核查清单及未匹配路径见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
