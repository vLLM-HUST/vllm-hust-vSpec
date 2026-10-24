# vSpec Adaptive NPU 验证报告

本页保留历史证据入口；完整测量、实现说明、配置、复现方法和失败记录见
[benchmark 历史原文](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/vspec-adaptive-narratives/snapshot/docs/vspec_adaptive_npu_report_20260902.md)。原文默认值属于当时环境，不代表当前安装配置。
GSM8K 质量差异、负收益、gamma-zero 小样本诊断及缺 checkpoint 限制原样保留；
本次迁移不新增正确性、性能或官方基线验收结论。

## 原始数据

以下固定链接指向实际 JSON/日志；配对范围、质量约束和计时边界须结合原文读取。

- [benchmark_results/draft_adaptive_full_async_b8_n2_max8_online_calibration.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_full_async_b8_n2_max8_online_calibration.json)
- [benchmark_results/draft_adaptive_full_async_b8_n2_max8_shared_target_params.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_full_async_b8_n2_max8_shared_target_params.json)
- [benchmark_results/draft_adaptive_g3_g4_b128_gsm8k_n200_max512_accuracy.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_g3_g4_b128_gsm8k_n200_max512_accuracy.json)
- [benchmark_results/draft_adaptive_gamma0_sync_transition_eager_b2_max32.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_gamma0_sync_transition_eager_b2_max32.json)
- [benchmark_results/draft_adaptive_gamma0_sync_transition_full_decode_async_b2_max12_v2.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_gamma0_sync_transition_full_decode_async_b2_max12_v2.json)
- [benchmark_results/draft_adaptive_online_latency_full_async_b128_n8_max32_fixed.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_online_latency_full_async_b128_n8_max32_fixed.json)
- [benchmark_results/draft_adaptive_online_latency_full_b128_gsm8k_n200_max512_fixed.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_online_latency_full_b128_gsm8k_n200_max512_fixed.json)
- [benchmark_results/draft_adaptive_position_ac_full_decode_async_b8_n8_max32_v9.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_position_ac_full_decode_async_b8_n8_max32_v9.json)
- [benchmark_results/draft_adaptive_targetonly_full_async_b8_n8_max32_v4.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_targetonly_full_async_b8_n8_max32_v4.json)
- [benchmark_results/draft_adaptive_targetonly_full_decode_async_b8_n8_max32.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_adaptive_targetonly_full_decode_async_b8_n8_max32.json)
- [benchmark_results/draft_fixed_g3_b128_gsm8k_n200_max512_accuracy.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/draft_fixed_g3_b128_gsm8k_n200_max512_accuracy.json)
- [benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy.json)
- [benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy_repeat2.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_adaptive_g2_g1_b1_metadata_event_b128_gsm8k_n200_max512_accuracy_repeat2.json)
- [benchmark_results/eagle_fixed_g2_metadata_event_b128_gsm8k_n200_max512_accuracy.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/eagle_fixed_g2_metadata_event_b128_gsm8k_n200_max512_accuracy.json)
- [benchmark_results/target_baseline_eager_b2_transition_max32.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/target_baseline_eager_b2_transition_max32.json)
- [benchmark_results/target_baseline_full_async_b8_n8_max32.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/target_baseline_full_async_b8_n8_max32.json)
- [benchmark_results/target_baseline_full_decode_async_b8_n8_max32.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/target_baseline_full_decode_async_b8_n8_max32.json)

下载及 SHA 核验见[原件说明](benchmark-originals.md)，当前使用方法见 [README](../README.md)。
