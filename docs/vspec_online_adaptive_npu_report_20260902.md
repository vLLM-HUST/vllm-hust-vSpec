# vSpec Online Adaptive NPU 报告（2026-09-02）

本页保留历史证据入口；完整测量、环境、启动方法、质量差异、失败及负收益已原样归档至
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2a97ad593c4fc3a6df051ec6dc3b78675a7a6225/archive/mod-targeted-20261023/vspec-legacy-narratives/snapshot/docs/vspec_online_adaptive_npu_report_20260902.md)。原文中的默认值和本地路径属于当时环境，
不代表当前安装配置；迁移不新增正确性、性能或官方基线验收结论。

## 原始数据

以下链接指向固定 benchmark 原件，保留原始版本与诊断身份。配置、配对范围和计时边界请结合历史原文读取。

- [benchmark_results/draft_fixed_g1_contemporary2_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g1_contemporary2_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_fixed_g2_contemporary2_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g2_contemporary2_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_fixed_g3_contemporary_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g3_contemporary_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_fixed_g4_contemporary2_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g4_contemporary2_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_online_ucb_confidence_highgamma_final_c4_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_online_ucb_confidence_highgamma_final_c4_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/draft_online_ucb_valid_c4_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_online_ucb_valid_c4_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_fixed_contemporary_full_b128_gsm8k_n200_max512.json)
- [benchmark_results/eagle_online_ucb_final_c2_full_b128_gsm8k_n200_max512.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_online_ucb_final_c2_full_b128_gsm8k_n200_max512.json)

下载与 SHA 核验见[原件说明](benchmark-originals.md)；当前安装和使用方法见 [README](../README.md)。
