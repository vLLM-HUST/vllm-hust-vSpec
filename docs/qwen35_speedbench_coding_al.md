# Qwen3.5-35B-A3B MTP2 SPEED-Bench coding acceptance length

本页为历史证据导航。[完整原文与复现方法](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/51604897aabb9a856b311fa58fc6749938d0378e/archive/mod-targeted-20261023/vspec-agentx-school/snapshot/docs/qwen35_speedbench_coding_al.md)已原样归档，保留全部配置、数值、失败与适用边界，不代表当前默认部署或新验收结论。

## 可核验原始数据

- [benchmark_results/qwen35_speedbench_al/20260928-044132-on/al_summary.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_speedbench_al/20260928-044132-on/al_summary.json)
- [benchmark_results/qwen35_speedbench_al/20260928-044132-on/dataset.sha256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_speedbench_al/20260928-044132-on/dataset.sha256)
- [benchmark_results/qwen35_speedbench_al/20260928-045302-off/al_summary.json](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_speedbench_al/20260928-045302-off/al_summary.json)
- [benchmark_results/qwen35_speedbench_al/20260928-045302-off/dataset.sha256](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/snapshot/benchmark_results/qwen35_speedbench_al/20260928-045302-off/dataset.sha256)

## 原件覆盖边界

以下成员未在本次固定的七份 vSpec capsule 清单内匹配；不作全组织永久丢失判定，不以摘要冒充逐请求原件，也不启动补测。

- `benchmark_results/qwen35_speedbench_al/20260928-044132-on/result.json`
- `benchmark_results/qwen35_speedbench_al/20260928-044132-on/bench.log`
- `benchmark_results/qwen35_speedbench_al/20260928-044132-on/server.log`
- `benchmark_results/qwen35_speedbench_al/20260928-044132-on/serve_cmd.txt`
- `benchmark_results/qwen35_speedbench_al/20260928-044132-on/bench_cmd.txt`
- `benchmark_results/qwen35_speedbench_al/20260928-045302-off/result.json`
- `benchmark_results/qwen35_speedbench_al/20260928-045302-off/bench.log`
- `benchmark_results/qwen35_speedbench_al/20260928-045302-off/server.log`
- `benchmark_results/qwen35_speedbench_al/20260928-045302-off/serve_cmd.txt`
- `benchmark_results/qwen35_speedbench_al/20260928-045302-off/bench_cmd.txt`

现存 AL 摘要和 dataset hash 不等于完整生成记录、服务日志或无回退证明。历史 thinking on/off 采样参数、原始计数与 AL 定义分别保存在完整原文和摘要中；本地参考值不冒称上游 golden AL。准备工具仍见 [`prepare_speedbench_coding.py`](../scripts/prepare_speedbench_coding.py) 和 [`measure_qwen35_speedbench_al.sh`](../scripts/measure_qwen35_speedbench_al.sh)。

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确绑定和缺失范围见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
