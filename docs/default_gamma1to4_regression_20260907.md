# vSpec 默认 Adaptive 回归结果

本页保留历史证据入口；完整测量、环境、启动方法、质量差异、失败及负收益已原样归档至
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2a97ad593c4fc3a6df051ec6dc3b78675a7a6225/archive/mod-targeted-20261023/vspec-legacy-narratives/snapshot/docs/default_gamma1to4_regression_20260907.md)。原文中的默认值和本地路径属于当时环境，
不代表当前安装配置；迁移不新增正确性、性能或官方基线验收结论。

## 原始数据

以下链接指向固定 benchmark 原件，保留原始版本与诊断身份。配置、配对范围和计时边界请结合历史原文读取。

- [benchmark_results/default_gamma1to4_regression_20260907/draft_adaptive_g1to4_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/draft_adaptive_g1to4_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/default_gamma1to4_regression_20260907/draft_adaptive_g1to4_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/draft_adaptive_g1to4_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/default_gamma1to4_regression_20260907/eagle_adaptive_g1to4_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/eagle_adaptive_g1to4_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/default_gamma1to4_regression_20260907/eagle_adaptive_g1to4_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/eagle_adaptive_g1to4_full_b128_gsm8k_n200_max512.log)
- [benchmark_results/default_gamma1to4_regression_20260907/target_baseline_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/target_baseline_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/default_gamma1to4_regression_20260907/target_baseline_full_b128_gsm8k_n200_max512.log](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/default_gamma1to4_regression_20260907/target_baseline_full_b128_gsm8k_n200_max512.log)

下载与 SHA 核验见[原件说明](benchmark-originals.md)；当前安装和使用方法见 [README](../README.md)。
