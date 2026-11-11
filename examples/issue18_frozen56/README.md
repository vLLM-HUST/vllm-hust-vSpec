# Issue #18 冻结 56 题负载上的新版优化栈复测

此入口复用 MMLU-Pro 冻结的 56 道正式题和 14 道不重叠 pilot 题，但**不是老师 2026-10-27 启动包的逐参数重放**。当前复测使用 Qwen3.5-35B-A3B、两张 910B2、BF16、TP=2/EP、最大并发 16、自动 KV（`gpu_memory_utilization=0.9`）、`FULL_DECODE_ONLY`、静态算子和本插件的优化后端；老师原始启动包为 `FULL_AND_PIECEWISE`、另一份 capture-size 列表，原生 MTP4 未开启本入口的本地 argmax 优化，candidate 最大深度为 6 且历史实际固定有效 gamma4。不能把两套参数的吞吐或加速比混称为同一实验。

正式题一次提交，每题最多输出 4000 token，自然停止；吞吐为正式输出 token 数除以正式生成墙钟时间，不含启动和预热。当前复测的原生 MTP4 与 vSpec 上限 8 参数在 [`v0.15.2-issue18-frozen56-auto-kv.json`](../../configs/v0.15.2-issue18-frozen56-auto-kv.json)，完整非路径运行开关在 [`v0.15.2-issue18-frozen56-env.json`](../../configs/v0.15.2-issue18-frozen56-env.json)。vSpec 用 2/4/6/8 在线档位，`archived-20261103` 策略从 8 开始并允许降档、回升。若要重放老师的原始参数，应从其 [launch-package](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a8e71b656f5f45d836ed2c72d2cad3dfe44457d7/reports/runtime-mod-review-20261027/launch-package) 单独启动并保留原始配置。

历史固定 KV 对照配置在 [`v0.15.2-issue18-frozen56.json`](../../configs/v0.15.2-issue18-frozen56.json)；自动 KV 配置相对它仅删除 `kv_cache_memory_bytes=15569256448`，保留 `gpu_memory_utilization=0.9`。若比较两种分配方式，必须让两臂使用同一 KV 口径；不要把一臂固定、一臂自动的吞吐相除。

先按[安装与硬件准备](../../docs/v0.15.1-reproduction.md)在独立 Python 环境中安装插件、匹配的 vLLM/Ascend 后端及算子，并保存安装器在**当前物理卡**上生成的 `runtime.env`。需另外准备本地模型、MMLU-Pro 测试和验证 parquet，以及 `vLLM-HUST/vllm-hust-benchmark@0ce4365692c335f31e2cbbb043d251e7f3d185ec` 的 `reports/runtime-mod-review-20261027/research/speculative-mmlu-pro/evidence/screen-manifest.json`。脚本会校验 parquet SHA-256、56/14 题 ID、各学科五个示例，并生成完整 Qwen chat template 提示词；数据和模型不随仓库发布。基础 vLLM 提交为 `762f85b311fbab0bcf8921dd216f5093cd58b9b8`，后端提交为 `4e57439e58ed3d78e675f9fd7b4614fb183c5394`，CANN 为 9.1.0。

在 CANN 环境已由系统加载、且两张分配的 NPU 空闲时，分别在全新的输出目录运行两臂；请把下面路径替换为本机路径。`SITE_PACKAGES` 是**包含** `vllm_hust_vspec` 和 `vllm_ascend` 两个目录的 Python site-packages。不要把凭据、题目或安装器的私有算子证明提交仓库。

输出根目录请使用短绝对路径；当前 CANN 静态编译器会把完整路径编码进单个生成文件名，过长会在首次启动时触发 `File name too long`。入口会在占用 NPU 前拒绝此类路径。脚本不指定物理卡号，使用宿主机或容器分配给进程的可见设备；安装器的 `runtime.env` 需在目标机器重新生成，不能把另一台机器的算子验证产物当作可移植资产。

```bash
VSPEC_REPO=/absolute/path/to/vllm-hust-vSpec
EVAL_PYTHON=/absolute/path/to/venv/bin/python
SITE_PACKAGES=/absolute/path/to/venv/lib/python3.11/site-packages
MODEL_DIR=/absolute/path/to/Qwen3.5-35B-A3B
TEST_PARQUET=/absolute/path/to/test-00000-of-00001.parquet
VALIDATION_PARQUET=/absolute/path/to/validation-00000-of-00001.parquet
FROZEN_MANIFEST=/absolute/path/to/screen-manifest.json
RUNTIME_ENV=/absolute/path/to/installer-output/runtime.env
RUN_ROOT=/absolute/path/to/private-results

"$EVAL_PYTHON" "$VSPEC_REPO/examples/issue18_frozen56/reproduce.py" \
  --arm native --test-parquet "$TEST_PARQUET" \
  --validation-parquet "$VALIDATION_PARQUET" --manifest "$FROZEN_MANIFEST" \
  --model-dir "$MODEL_DIR" \
  --engine-config "$VSPEC_REPO/configs/v0.15.2-issue18-frozen56-auto-kv.json" \
  --env-profile "$VSPEC_REPO/configs/v0.15.2-issue18-frozen56-env.json" \
  --installed-site-packages "$SITE_PACKAGES" --chat-template \
  --output "$RUN_ROOT/native-r1"

"$EVAL_PYTHON" "$VSPEC_REPO/examples/issue18_frozen56/reproduce.py" \
  --arm vspec --test-parquet "$TEST_PARQUET" \
  --validation-parquet "$VALIDATION_PARQUET" --manifest "$FROZEN_MANIFEST" \
  --model-dir "$MODEL_DIR" \
  --engine-config "$VSPEC_REPO/configs/v0.15.2-issue18-frozen56-auto-kv.json" \
  --env-profile "$VSPEC_REPO/configs/v0.15.2-issue18-frozen56-env.json" \
  --installed-site-packages "$SITE_PACKAGES" \
  --runtime-env-file "$RUNTIME_ENV" --chat-template \
  --output "$RUN_ROOT/vspec-r1"
```

这两条命令只做冻结题单原有的 14 题、每题 16 token 短预热；但引擎及后端参数仍属于上述新版优化栈。若要检验首次长生成算子成本，可以**给两臂同时**加 `--prime-pilot16x1024`：额外用 16 个未计分 pilot 请求、每题最多 1024 token，在正式计时前预热。它会单独写出 `prime-summary.json`，但**不是** issue #18 原始冻结协议；不能与未做同样额外预热的原生结果直接比较。每次使用新输出目录，脚本会把 Triton、TorchInductor、ACL 等编译缓存隔离到该目录；不要在两臂运行期间并发占用同一对 NPU。

当前 vSpec 环境配置还启用了 `VLLM_ASCEND_Q35_GDN_PRECOMPILE_WIDTHS=1`：后端在首次符合条件的短预热请求中，针对本机真实的张量布局预编译动态档位和尾部路径可能需要的 Triton 特化，不执行额外的模型生成。它不会读取题目内容，**会增加短预热耗时**；正式输出 token/s 仍不含这段时间。新安装插件与后端 wheel、复用未改动原生算子产物的空缓存配对轮测得 vSpec 964.01、原生 MTP4 802.53 token/s（1.201×）。随后运行完整安装器重新构建并验证算子，再以空缓存配对测得 936.19 对 731.77 token/s（1.279×）；原生第二轮吞吐低约 8.8%，不能只引用较低基线，详见[复测记录](../../docs/v0.15.2-issue18-cache-replay.md)。两轮都不是“空机器直接 pip install 即运行”，基础依赖、CANN、模型与数据仍需准备。若比较首轮端到端耗时，应把服务启动、短预热和正式题全部计入，而不是只看正式吞吐。

要比较“首次编译”和“重用已有编译产物”，可在第二轮给每一臂分别加 `--reuse-cache-from "$RUN_ROOT/native-r1"` 或 `--reuse-cache-from "$RUN_ROOT/vspec-r1"`，并指定新的输出目录。脚本会复制该臂上一轮的缓存，不会修改原结果；收据里的 `reuse_cache_from` 用于标明这是**复用缓存实验**，不能报告成全新冷缓存成绩。两臂都应采用相同缓存口径、相同题单与预热规则，再计算配对倍率。该选项不增加计时前的长生成 pilot，但保留之前运行编译出的算子产物。

最终比较使用两份 `summary.json` 的 `output_tokens_per_second`。这里比较的是完整安装栈：vSpec 臂开启本插件的优化后端路径，原生臂关闭 vSpec 动态选择，因此倍率不能单独归因于档位策略。56 题的正确数只用于诊断，不能代替 MMLU-Pro 全量质量结论。

本地安装包在全新缓存与复用自身缓存两种口径下的聚合观测和边界见[编译缓存复测记录](../../docs/v0.15.2-issue18-cache-replay.md)。

2026-11-11 使用新的虚拟环境、完整 `install.sh` 入口重新构建后端和全部原生算子后，再以自动 KV、空编译缓存和上述短 pilot 单独复测 vSpec：56/56 完成、零抢占，149,777 输出 token / 153.45 秒，**976.05 输出 token/s**；服务就绪 700.08 秒。完整的脱敏配置哈希、安装边界和原始聚合结果见[结果文件](result-20261111-new-env-vspec-auto-kv.json)。这次未在同一新环境配对原生 MTP4，不能从该单臂结果计算新的加速比；39/56 的小样本正确数也不代表全量准确率。
