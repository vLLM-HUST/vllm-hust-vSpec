# vSpec 动态 gamma 优化报告

本页保留历史证据入口；完整测量、实现说明、配置、复现方法和失败记录见
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/vspec-adaptive-narratives/snapshot/docs/dynamic_gamma_gsm8k_b128_optimization_report.md)。原文默认值属于当时环境，不代表当前安装配置。
GSM8K 质量差异、负收益、gamma-zero 小样本诊断及缺 checkpoint 限制原样保留；
本次迁移不新增正确性、性能或官方基线验收结论。

## 原始数据

以下固定链接指向实际 JSON/日志；配对范围、质量约束和计时边界须结合原文读取。

- [benchmark_results/draft_fixed_g2_contemporary2_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g2_contemporary2_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_online_dynamic_goal_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_online_dynamic_goal_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_online_dynamic_goal_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_online_dynamic_goal_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_dynamic_goal_full_b128_gsm8k_n200_max512.log)

下载及 SHA 核验见[原件说明](benchmark-originals.md)，当前使用方法见 [README](../README.md)。
