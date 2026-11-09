# Qwen3.5 三数据集配对评测入口

这个目录提供可公开审计的**入口脚本**，不附带模型、数据集、私人配置、原始实验结果或历史环境的绝对路径。[本地复测记录](../../docs/v0.15.1-local-retest.md)另行列出实际参数和脱敏聚合结果。入口生成同口径的原生 MTP 与动态 vSpec 服务配置，顺序运行两臂，记录请求用时与输出 token，并把评分同吞吐分开记录。原生候选上限可设为 4、6 或 8；四组对照分别建立同参数的原生 setup，并将 vSpec 结果与各原生结果并列汇总。所有计划、日志和原始结果必须存到操作者的私有目录；模型与数据集资产可以由只读共享存储提供。

这不是“安装插件即复现报告数值”的保证。复现所需的底层版本、NPU、算子、judge 服务及证明资产必须另行核验。脚本不会构造算子证明资产，也不提供 HLE/FrontierScience 的评审模型服务。模型级端点能读取实际 worker、静态编译与算子状态，但这些观察值不能代替数值正确性、完整评分或独立审计。后测算子的折叠计数可能包含启动期 graph capture，只能证明同一 worker 进程观察到算子执行，不能单独证明计时窗口内发生了命中；正式声明需另外提供计时前后计数差或独立 profiler 证据。模型权重未逐字节哈希，setup 仅绑定模型目录路径，模型修订与权重哈希需由操作者另行记录。

## 准备条件

- 使用锁定版本的干净 Python 环境、vLLM、vllm-ascend、CANN、torch-npu、插件 wheel 和匹配的 custom OPP。两臂可以分别使用隔离环境；空间受限时，也可以在同一个干净环境中顺序运行，给原生臂指定独立的只读原生后端覆盖层。此时必须用 worker 记录证明原生子进程未导入 vSpec，且两臂确实使用各自指定的后端和 OPP。`run_with_backend` 会检查实际导入路径；仅复制旧工作目录或设置历史 `SOURCE` 不符合此条件。
- 同一份有固定 revision/hash 的 Qwen3.5 模型、同一对物理 NPU、BF16、同一份共享服务配置。原生服务的 `VLLM_PLUGINS` 为 `ascend,vspec_eval`；动态服务为 `ascend,vspec,vspec_eval`。两臂都启用插件包中独立的 Qwen3.5 MTP KV/Mamba 兼容启动钩子；原生服务的子进程不应导入 `vllm_hust_vspec`。
- 一个干净的评测客户端 Python 环境，安装并固定对应版本的 `hust-eval-client-mmlu-pro`、`hust-eval-client-hle-verified`、`hust-eval-client-frontierscience`。按各客户端文档准备受许可的数据资产、solver/judge 配置和冻结计划。HLE-Verified 用 Gold；FrontierScience 计时覆盖 olympiad 与 research，质量只按 olympiad 的评分定义。切勿将 MMLU 或 SWE-bench Verified 代替相应数据集。
- 插件安装包内的独立原生编译适配/观察类与通用评测端点。原生子进程仅将安装包的独立 `evaluation_static`、`evaluation` 文件目录加入 Python 模块搜索路径，端点与原生 worker 均不导入 vSpec 插件；`VLLM_PLUGINS` 只显式加载 Ascend 和 `vspec_eval`。动态子进程加载 Ascend、vSpec 与 `vspec_eval`。不要用私有旧 `SOURCE` 覆盖这两个目录。
- 动态 worker 所需的 proof/构建产物路径由操作者在私有 `dynamic-env.json` 中提供。`*_DIR`、`*_ROOT`、`*_PATH` 键必须指向存在的绝对路径。请先执行插件的发行校验与目标机器上的模型级验收，确认 wheel/sdist、后端覆盖层、算子和 worker 的版本、路径与哈希吻合。

例如 RGDR state pipeline 应从**本次安装的 wheel**执行 `vllm-hust-vspec-build-operator --operator rgdr-state-pipeline --cann-root <CANN_ROOT> --destination <PRIVATE_RGDR_DIR>`，再执行 `vllm-hust-vspec-validate-rgdr-state-pipeline --component-dir <PRIVATE_RGDR_DIR> --backend-package <VENV_SITE_PACKAGES>`；其它算子按 [发布版复现文档](../../docs/v0.15.1-reproduction.md)构建/校验。这里的组件验证只证明安装资产符合约束，不能替代实际 Qwen3.5 三数据集的模型级运行证据。

`common-config.json` 是用户自己冻结的 vLLM 顶层配置（JSON 对象，可包含其他所需参数），至少包含 `dtype`, `tensor_parallel_size`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `enable_prefix_caching`, `kv_cache_memory_bytes`, `compilation_config`, `speculative_config`。这里要求 `dtype=bfloat16`、TP=2、`kv_cache_memory_bytes` 为正整数或 `null`（自动分配）、`speculative_config.method=mtp`；不要把 `model`、`host`、`port`、`served_model_name`、`worker_extension_cls` 或 `num_speculative_tokens` 塞进共享文件。服务配置生成器只允许两臂在投机深度（原生可选 4/6/8、动态容量 8）和 worker 类上不同；数据集可用各自的长度/多模态配置，但**同一数据集的两臂必须用同一共享配置**。动态容量 8 是上限，不代表每轮都固定选 8；实际选择仍需独立 worker 证据核验。

`common-env.json` 和 `dynamic-env.json` 都是私有 JSON 字符串映射。前者可放两臂共同的设备可见性、离线模型、静态编译设置，其中 `HUST_VSPEC_CAPTURE_SAFE_STATIC="1"` 与 `HUST_VSPEC_HOT_STATIC_SHAPES="14,42,70,98,126"` 是本配对入口要求的共同编译设置；还可放六个固定取值的 vLLM 共同开关：`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`、`VLLM_NO_USAGE_STATS=1`、`VLLM_SERVER_DEV_MODE=0`、`VLLM_USE_AOT_COMPILE=0`、`VLLM_USE_BREAKABLE_CUDAGRAPH=0`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`。其他 `VLLM_*` 不能混入共同环境。后者放动态臂的 `VSPEC_`、`HUST_VSPEC_`、`HUST_ISSUE18_` 配置与 proof 路径，以及显式允许的 Qwen3.5 后端开关。不要在这两个文件中放密码、API token 或旧环境 `PYTHONPATH`。服务子进程只继承必要的系统/NPU 环境变量，并剔除常见凭据变量；需要的额外非敏感设置请显式写入 common-env。示例自行生成保存在私有 setup 中的随机端点令牌，为每臂新建隔离缓存目录，并在计时前调用受保护的缓存清理路由；不会预存答案或 KV。所有由评测入口设置的 vLLM、Hugging Face、Ascend、编译器及绘图库缓存路径都指向本次私有运行目录，不沿用旧实验或共享区缓存。

本入口只允许在 `dynamic-env.json` 中显式设置列入白名单的 `VLLM_ASCEND_Q35_*` 开关；`common-env.json` 不接受这些键，原生臂使用所选原生后端的默认值。这些开关也不会从当前 shell 隐式继承。因此，同一共享服务配置和公共编译设置不表示两臂的 Qwen3.5 后端优化相同。历史三数据集对照中的原生与动态实现也有逐臂差异；复核历史配置时应在私有启动记录中核对各臂实际开关、后端和算子，不能把整体吞吐差异单独归因于动态控制器。`VLLM_ALLOW_INSECURE_SERIALIZATION=0` 由入口对两臂强制设置，不需要写进私有环境文件。

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
  --native-evaluation-dir /path/to/fresh/native-venv/lib/pythonX.Y/site-packages/vllm_hust_vspec/evaluation \
  --dynamic-evaluation-dir /path/to/fresh/dynamic-venv/lib/pythonX.Y/site-packages/vllm_hust_vspec/evaluation \
  --served-model paired-solver --port 18080 --native-capacity 4
```

若原生 `vllm_ascend` 是单独的后端覆盖层，`--native-backend-package` 指向该覆盖层，`--native-evaluation-dir` 仍指向**同一原生 Python 环境安装的插件 wheel**；两者不必是同一目录。原生静态 worker 位于该 evaluation 目录的同级 `evaluation_static`，由入口核验后加入子进程模块路径。

需要 MTP6/MTP8 时，分别以 `--native-capacity 6` 或 `8` 创建新的私有 setup 和运行目录；若使用自动 KV，将共享配置中的 `kv_cache_memory_bytes` 设为 JSON `null`。四组均不预热时不要传 `--warmup-selection`。每个 setup 仍只运行一组原生配置与 vSpec；跨 setup 的四组表格由操作者依据冻结题单和评分汇总生成，不要把不同题单或不同 KV 策略合并。

HLE-Verified 与 FrontierScience 需分别执行一次 `prepare`，各自使用本数据集冻结的配置及 judge/solver plan。两个客户端的 `contract.json` 中 solver URL/模型名必须和上述 loopback 地址/别名一致，否则 `run` 会拒绝。判断服务可在不同端口运行，但应由操作者按客户端冻结的 judge 配置单独管理。若计划由不同客户端版本生成，应先重新冻结并记录版本/哈希。

可选的 `--warmup-selection` 在同一个新启动的服务上先运行**非计时** smoke，再获取计时前 worker 元数据、清理 prefix/encoder/多模态缓存，最后执行正式计时。未提供该选项时不运行 smoke；两臂应显式使用同一份私有 manifest，`compare` 会核对其文件哈希及题目哈希。smoke 写入独立的 `warmup/`、`warmup.log` 和 `warmup-receipt.json`，正式请求只写入 `solver/`；客户端不使用 `--resume`，smoke 失败或缓存清理未获完整确认时正式计时不会开始。warmup 时间、请求与 token 均不计入正式吞吐，答案和 KV 不从先前输出恢复。

私有 warmup manifest 是 JSON 对象，包含 `schema_version: 1`、`dataset`、`source_plan_contract_sha256`（本次 `plan/contract.json` 的文件 SHA-256）、`count`、`case_ids` 和 `case_ids_sha256`（按本脚本 `canonical_sha256` 计算）。入口核验题目唯一性、数量、哈希和同一冻结 plan 的归属。MMLU-Pro 客户端只支持 `--limit N`，因此 warmup 必须是正式 200 题计划的前 N 题；HLE/FrontierScience 使用逐题 `--case-id`，其 warmup 题可以不在正式评分选单内，但必须属于同一冻结 plan 的完整任务清单。manifest 与题目 ID 只保存在操作者私有目录，不提交到公开仓库。

## 顺序运行、评分与对比

下面展示一个数据集的命令形状。先不带 `--execute` 检查命令与题数，再加上它执行；每一臂使用新 tag 与全新输出目录。两臂不同时启动服务，避免争用同卡。入口先通过私有令牌从两个 worker 收集只读元数据并清理 prefix/encoder/多模态缓存，计时后再取 worker 和算子证据；任何一步失败都不生成有效对照。计时从客户端开始到结束，**不包含模型启动/编译、证据读取与 judge**，但包含客户端开销。输出吞吐是实际生成 `completion_tokens / wall_seconds`；自然输出长度可能不同，它不是固定工作量加速比。

```bash
SETUP="$PRIVATE_ROOT/pair-mmlu/setup.json"
PLAN="$PRIVATE_ROOT/mmlu/plan200"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm native --dataset mmlu-pro --plan "$PLAN" \
  --warmup-selection "$PRIVATE_ROOT/mmlu/warmup.json" --tag run1
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm native --dataset mmlu-pro --plan "$PLAN" \
  --warmup-selection "$PRIVATE_ROOT/mmlu/warmup.json" --tag run1 --execute
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" run --setup "$SETUP" \
  --arm dynamic --dataset mmlu-pro --plan "$PLAN" \
  --warmup-selection "$PRIVATE_ROOT/mmlu/warmup.json" --tag run1 --execute

"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" score --setup "$SETUP" \
  --plan "$PLAN" --run-dir "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/native/run1"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" score --setup "$SETUP" \
  --plan "$PLAN" --run-dir "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/dynamic/run1"
"$CLIENT_PYTHON" "$EXAMPLE_DIR/paired.py" compare \
  --native-run "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/native/run1" \
  --dynamic-run "$PRIVATE_ROOT/pair-mmlu/runs/mmlu-pro/dynamic/run1" \
  --output "$PRIVATE_ROOT/pair-mmlu/comparison.json"
```

HLE/FrontierScience 的 `run` 命令还需 `--selection /path/to/selection.json`。两个 solver run 完成后，按冻结配置在独立服务上启动 judge，再执行各自的 `score`。`score` 和 `compare` 会重验计时请求文件及计划题目清单的哈希。`compare` 核对同一 plan、题目清单、共享服务配置、评分范围和规则；HLE/FrontierScience 把未决评分按最不利情况计算质量下降界，不把未决题自动判为正确。`performance_quality_scope_pass` 仅表示吞吐、质量与样本范围三项通过：MMLU-Pro/HLE-Verified 至少 200 题，FrontierScience 至少 160 题且覆盖完整双轨计划；任一项不满足时仍打印诊断结果，但非零退出。它不代表完整发布验收，后者仍需独立算子数值证明并运行 `scripts/verify_model_acceptance.py`。MMLU-Pro 按客户端的完整总结评分；截断题仍按规则计入分母。

运行日志、计划、选题、请求记录、评分和配置会包含模型路径或题目内容，**不要上传到公开仓库或共享目录**。脚本在正式运行时从 `npu-smi` 记录逻辑 0/1 对应的两块物理 910B2，`compare` 仅在两臂映射一致时继续；但仍须独立保存设备租用和真实 worker/算子证据。公开报告仅可发布脱敏后的聚合结果，并应附版本/哈希、硬件分配、两臂运行时 worker/算子证明、客户端/模型/计划修订及完整计分口径。没有这些证据，就只能称为本地诊断，不应宣称复现了既有吞吐或正确率。
