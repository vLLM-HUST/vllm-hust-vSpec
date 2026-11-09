# Qwen3.5 三数据集配对评测入口

这个目录提供可公开审计的**入口脚本**，不附带模型、数据集、私人配置、实验结果或历史环境的绝对路径。它生成同口径的原生 MTP4 与动态 vSpec 服务配置，顺序运行两臂，记录请求用时与输出 token，并把评分同吞吐分开记录。所有计划、日志和结果必须存到操作者的私有目录；模型与数据集资产可以由只读共享存储提供。

这不是“安装插件即复现报告数值”的保证。复现所需的底层版本、NPU、算子、模型级 worker、judge 服务及证明资产必须另行核验。脚本本身不生成后端算子证明，不检查真实物理 NPU 是否为同一对，也不提供 HLE/FrontierScience 的评审模型服务。`compare` 输出仍标明独立运行时与算子证明缺失；不能把它当成完整模型级验收凭证。

## 准备条件

- 两个**全新、隔离**的 Python 环境，分别装原生后端及目标 vSpec 后端；两者按同一锁定版本安装 vLLM、vllm-ascend、CANN、torch-npu、插件 wheel 和匹配的 custom OPP。`run_with_backend` 会确认当前 Python 实际导入的是传入的后端包，并把对应 OPP 配给子进程；仅复制旧工作目录或设置历史 `SOURCE` 不符合此条件。
- 同一份有固定 revision/hash 的 Qwen3.5 模型、同一对物理 NPU、BF16、同一份共享服务配置。原生服务的 `VLLM_PLUGINS` 只启用 Ascend；动态服务启用 Ascend 与 vSpec。两臂都启用插件包中独立的 Qwen3.5 MTP KV/Mamba 兼容启动钩子；原生服务的子进程不应导入 `vllm_hust_vspec`。
- 一个干净的评测客户端 Python 环境，安装并固定对应版本的 `hust-eval-client-mmlu-pro`、`hust-eval-client-hle-verified`、`hust-eval-client-frontierscience`。按各客户端文档准备受许可的数据资产、solver/judge 配置和冻结计划。HLE-Verified 用 Gold；FrontierScience 计时覆盖 olympiad 与 research，质量只按 olympiad 的评分定义。切勿将 MMLU 或 SWE-bench Verified 代替相应数据集。
- 一个**可导入**的原生 worker 观察类，用 `--native-worker` 传入。此类应只观测原生实现，不启用 vSpec 插件/策略。没有它可以做启动诊断，但 `compare` 会拒绝将结果当作可比对实验。当前示例不打包该模型级观察类或历史运行时 metadata/reset 端点；发布前仍需补齐并做审计。
- 动态 worker 所需的 proof/构建产物路径由操作者在私有 `dynamic-env.json` 中提供。`*_DIR`、`*_ROOT`、`*_PATH` 键必须指向存在的绝对路径。请先执行插件的发行校验与目标机器上的模型级验收，确认 wheel/sdist、后端覆盖层、算子和 worker 的版本、路径与哈希吻合。

例如 RGDR state pipeline 应从**本次安装的 wheel**执行 `vllm-hust-vspec-build-operator --operator rgdr-state-pipeline --cann-root <CANN_ROOT> --destination <PRIVATE_RGDR_DIR>`，再执行 `vllm-hust-vspec-validate-rgdr-state-pipeline --component-dir <PRIVATE_RGDR_DIR> --backend-package <VENV_SITE_PACKAGES>`；其它算子按 [发布版复现文档](../../docs/v0.15.1-reproduction.md)构建/校验。这里的组件验证只证明安装资产符合约束，不能替代实际 Qwen3.5 三数据集的模型级运行证据。

`common-config.json` 是用户自己冻结的 vLLM 顶层配置（JSON 对象，可包含其他所需参数），至少包含 `dtype`, `tensor_parallel_size`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, `kv_cache_memory_bytes`, `compilation_config`, `speculative_config`。这里要求 `dtype=bfloat16`、TP=2、显式正数 KV byte 预算、`speculative_config.method=mtp`；不要把 `model`、`host`、`port`、`served_model_name`、`worker_extension_cls` 或 `num_speculative_tokens` 塞进共享文件。服务配置生成器只允许两臂在投机深度（原生 4、动态容量 8）和 worker 类上不同；数据集可用各自的长度/多模态配置，但**同一数据集的两臂必须用同一共享配置**。动态容量 8 是上限，不代表每轮都固定选 8；实际选择仍需独立 worker 证据核验。

`common-env.json` 和 `dynamic-env.json` 都是私有 JSON 字符串映射。前者可放两臂共同的设备可见性、离线模型、静态编译设置；后者只放动态臂的 `VSPEC_`、`HUST_VSPEC_` 或 `HUST_ISSUE18_` 配置与 proof 路径。不要在这两个文件中放密码、API token 或旧环境 `PYTHONPATH`。示例会为每臂新建独立缓存目录，但这不等于证明设备、编译和请求缓存状态完全相同。

## 冻结题目与准备配对配置

先按各客户端的正式 `prepare` 流程生成冻结计划。MMLU-Pro 应使用客户端正式 plan，不能把 `run --limit` 的 smoke 结果当成可评分的 200 题。下面的变量均由操作者在自己的私有环境中设置；不要提交它们的展开值。

```bash
EXAMPLE_DIR=/path/to/installed/source/examples/qwen35_three_datasets
CLIENT_PYTHON=/path/to/fresh/client-venv/bin/python
PRIVATE_ROOT=/path/to/private/evaluation

# MMLU-Pro: 从完整计划派生可由客户端验证的 200 题子计划。
"$CLIENT_PYTHON" "$EXAMPLE_DIR/select_cases.py" \
  --dataset mmlu-pro --plan "$PRIVATE_ROOT/mmlu/full-plan" \
  --output "$PRIVATE_ROOT/mmlu/plan200" --count 200 --seed published-seed

# HLE-Verified: 从 Gold 计划中选 200 题。FrontierScience 省略 --count 可跑冻结计划全量。
"$CLIENT_PYTHON" "$EXAMPLE_DIR/select_cases.py" \
  --dataset hle-verified --plan "$PRIVATE_ROOT/hle/plan" \
  --output "$PRIVATE_ROOT/hle/selection200.json" --count 200 --seed published-seed
"$CLIENT_PYTHON" "$EXAMPLE_DIR/select_cases.py" \
  --dataset frontierscience --plan "$PRIVATE_ROOT/frontier/plan" \
  --output "$PRIVATE_ROOT/frontier/selection-full.json" --seed published-seed

"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" prepare \
  --private-root "$PRIVATE_ROOT/pair-mmlu" \
  --model-dir /path/to/read-only/model \
  --common-config "$PRIVATE_ROOT/mmlu/common-config.json" \
  --common-env "$PRIVATE_ROOT/common-env.json" \
  --dynamic-env "$PRIVATE_ROOT/dynamic-env.json" \
  --native-python /path/to/fresh/native-venv/bin/python \
  --dynamic-python /path/to/fresh/dynamic-venv/bin/python \
  --client-python "$CLIENT_PYTHON" \
  --native-backend-package /path/to/fresh/native-venv/lib/pythonX.Y/site-packages \
  --dynamic-backend-package /path/to/fresh/dynamic-venv/lib/pythonX.Y/site-packages \
  --native-worker your_installed_native_observer.NativeWorker \
  --served-model paired-solver --port 18080
```

HLE-Verified 与 FrontierScience 需分别执行一次 `prepare`，各自使用本数据集冻结的配置及 judge/solver plan。两个客户端的 `contract.json` 中 solver URL/模型名必须和上述 loopback 地址/别名一致，否则 `run` 会拒绝。判断服务可在不同端口运行，但应由操作者按客户端冻结的 judge 配置单独管理。若计划由不同客户端版本生成，应先重新冻结并记录版本/哈希。

## 顺序运行、评分与对比

下面展示一个数据集的命令形状。先不带 `--execute` 检查命令与题数，再加上它执行；每一臂使用新 tag 与全新输出目录。两臂不同时启动服务，避免争用同卡。计时从客户端开始到结束，**不包含模型启动/编译与 judge**，但包含客户端开销。输出吞吐是实际生成 `completion_tokens / wall_seconds`；自然输出长度可能不同，它不是固定工作量加速比。

```bash
SETUP="$PRIVATE_ROOT/pair-mmlu/setup.json"
PLAN="$PRIVATE_ROOT/mmlu/plan200"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm native --dataset mmlu-pro --plan "$PLAN" --tag run1
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm native --dataset mmlu-pro --plan "$PLAN" --tag run1 --execute
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm dynamic --dataset mmlu-pro --plan "$PLAN" --tag run1 --execute

"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" score --setup "$SETUP" \
  --plan "$PLAN" --run-dir "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/native/run1"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" score --setup "$SETUP" \
  --plan "$PLAN" --run-dir "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/dynamic/run1"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" compare \
  --native-run "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/native/run1" \
  --dynamic-run "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/dynamic/run1" \
  --output "$PRIVATE_ROOT/pair-mmlu/comparison.json"
```

HLE/FrontierScience 的 `run` 命令还需 `--selection /path/to/selection.json`。两个 solver run 完成后，按冻结配置在独立服务上启动 judge，再执行各自的 `score`。`compare` 核对同一 plan、题目清单、共享服务配置、评分范围和规则；HLE/FrontierScience 把未决评分按最不利情况计算质量下降界。它只给观察值，不把未决题自动判为正确。若出现客户端失败、不完整请求、评分范围不一致或非正的基线吞吐，会拒绝对比。MMLU-Pro 按客户端的完整总结评分；截断题仍按规则计入分母。

运行日志、计划、选题、请求记录、评分和配置会包含模型路径或题目内容，**不要上传到公开仓库或共享目录**。脚本在正式运行时从 `npu-smi` 记录逻辑 0/1 对应的两块物理 910B2，`compare` 仅在两臂映射一致时继续；但仍须独立保存设备租用和真实 worker/算子证据。公开报告仅可发布脱敏后的聚合结果，并应附版本/哈希、硬件分配、两臂运行时 worker/算子证明、客户端/模型/计划修订及完整计分口径。没有这些证据，就只能称为本地诊断，不应宣称复现了既有吞吐或正确率。
