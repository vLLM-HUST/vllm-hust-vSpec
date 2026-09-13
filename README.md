# vllm-hust-vSpec

`vllm-hust-vSpec` 是面向 vLLM-HUST + vLLM-Ascend-HUST 的独立
`vllm.general_plugins` 投机解码插件。插件支持动态 gamma，并提供已验证的 Draft 和 EAGLE 投机解码方案、模型配置、Eager/Graph 启动参数以及 Ascend方法专属优化。

Extension Manager ID：`org.vllm-hust.vspec`。插件遵循 Manifest
`0.2-experimental` 的 `in_process_plugin` 边界，同时注册
`vllm.general_plugins` 运行入口和 `vllm_hust.extension_bundles` 静态发现入口。

当前物理迁移范围、干净宿主 NPU 结果和剩余 ABI 边界见
[`docs/migration_status.md`](docs/migration_status.md)。

## 支持范围

| 启动预设 | 方法 | Target | Drafter |
|---|---|---|---|
| `draft` | `draft_model` | Qwen2.5-14B-Instruct | Qwen2.5-0.5B-Instruct |
| `eagle` | `eagle` | Qwen2.5-14B-Instruct | Eagle-Qwen2.5-14B-Instruct |
| `eagle-relaxed` | `eagle` | Qwen2.5-14B-Instruct | Eagle-Qwen2.5-14B-Instruct |

`eagle-relaxed` 是显式性能/质量折中预设，会使用 top-K relaxed acceptance，
输出不保证与严格 greedy EAGLE 一致。普通 `eagle` 始终默认 top-1 严格验收。

插件还提供 **vSpec Adaptive** 闭环控制。串行 Draft 和 EAGLE 启动时
默认启用无需离线标定的 `online` 策略；可用 `--no-adaptive-speculation` 显式关闭。
`profile` 策略使用离线 forward 延迟
模型和在线逐位置验收率；`online` 策略直接从当前服务的已提交 token 和非重叠
完成间隔学习，不需要预先执行硬件标定。两种策略都能逐 decode step 改变
batch-uniform 投机预算；online 还可叠加当前轮逐请求置信停止。配置与当前边界见
[`docs/vspec_adaptive.md`](docs/vspec_adaptive.md)。
同轮固定/动态 NPU A/B、正确率和失败方案见
[`docs/vspec_adaptive_npu_report_20260902.md`](docs/vspec_adaptive_npu_report_20260902.md)。
无离线 profile 的在线策略实现和 NPU 对比见
[`docs/vspec_online_adaptive_npu_report_20260902.md`](docs/vspec_online_adaptive_npu_report_20260902.md)。
最新同卡 GSM8K B128 `1.518x` 恢复验证、工程模板 GSM8K 与 ARC-Easy 结果见
[`docs/gsm8k_b128_recovery_and_arc_easy_20260913.md`](docs/gsm8k_b128_recovery_and_arc_easy_20260913.md)。
离线 profile 可用 `vllm-hust-vspec-profile` 自动采集固定 gamma component
stats、选择 batch policy 并保存完整 measurement manifest。

默认开启依据当前 Qwen2.5 B128 的端到端结果；其他模型、batch 和数据分布仍应以
自身吞吐测试为准。在线控制器会自适应 gamma，但不保证每一种负载都优于固定 gamma。

## 目录

```text
vllm-hust-vSpec/
├── .github/workflows/
│   ├── extension-ci.yml
│   └── release.yml
├── LICENSE
├── configs/
│   ├── qwen25-14b-05b.toml
│   ├── qwen25-14b-05b-arc-easy.toml
│   ├── qwen25-14b-eagle.toml
│   ├── qwen25-14b-eagle-arc-easy.toml
│   ├── qwen25-14b-eagle-relaxed.toml
├── profiles/
│   ├── qwen25_draft_gsm8k_graph_b128_g4.json
│   └── qwen25_eagle_gsm8k_graph_b128.json
├── src/vllm_hust_vspec/
│   ├── _version.py
│   ├── backends/
│   │   ├── draft.py
│   │   ├── eagle.py
│   │   ├── eagle_draft.py
│   │   ├── eagle_graph.py
│   │   ├── eagle_host.py
│   │   ├── eagle_metadata.py
│   │   ├── eagle_rejection.py
│   │   ├── eagle_runtime.py
│   │   └── eagle_target.py
│   ├── adaptive/
│   │   ├── controller.py
│   │   ├── entropy.py
│   │   ├── online.py
│   │   ├── profile.py
│   │   └── runtime.py
│   ├── models/
│   │   └── qwen2_eagle.py
│   ├── manifests/
│   │   ├── __init__.py
│   │   └── vllm-hust-extension-v0.2.json
│   ├── cli.py
│   ├── compatibility.py
│   ├── config.py
│   ├── model_store.py
│   └── patches.py
├── scripts/verify_release.py
├── tests/
│   ├── test_extension_lifecycle.py
│   ├── test_launcher.py
│   ├── test_management.py
│   └── test_manifest.py
├── install.sh
├── manage.sh
├── pyproject.toml
├── release.sh
├── run.sh
└── uninstall.sh
```

## 安装

统一管理入口默认安装当前版本 wheel；需要继续开发源码时使用 editable 模式：

```bash
cd /root/data/vllm-hust-vSpec
./manage.sh install
./manage.sh install --editable
./manage.sh install --enable
```

`./install.sh` 等价于 `./manage.sh install --editable`。安装会注册名为 `vspec` 的
`vllm.general_plugins` entry point 和 ID 为 `org.vllm-hust.vspec` 的
`vllm_hust.extension_bundles` entry point。默认安装不会自动启用；`--enable` 会在
Manager 完成静态发现和兼容性检查后显式启用。

`manage.sh install` 和 `install.sh` 还会自动准备两个默认 Drafter：

| 方法 | 默认仓库 | 默认目录名 |
|---|---|---|
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen2.5-0.5B-Instruct` |
| EAGLE | `Zjcxy-SmartAI/Eagle-Qwen2.5-14B-Instruct` | `Eagle-Qwen2.5-14B-Instruct` |

安装器先检查环境变量、已有登记和 `/data/shared-models` 下的模型，并校验
`config.json`、architecture、非空权重及 Draft tokenizer；只在没有可用副本时通过
`huggingface_hub` 下载已验证 revision。结果写入
`${XDG_CONFIG_HOME:-$HOME/.config}/vllm-hust-vspec/models.json`。下载根目录优先使用
可写的 `/data/shared-models`，否则使用
`${XDG_DATA_HOME:-$HOME/.local/share}/vllm-hust-vspec/models`。

```bash
# 自定义下载目录和登记文件
./manage.sh install --editable \
  --model-dir /models/vspec \
  --model-registry /etc/vllm-hust-vspec/models.json

# 离线环境只检测，不下载；缺少任一模型时安装失败
./manage.sh install --editable --no-model-download

# 单独补做模型准备
./manage.sh models
```

`HUST_VSPEC_DRAFT_MODEL`、`HUST_VSPEC_EAGLE_MODEL`、`HUST_VSPEC_MODEL_DIR` 和
`HUST_VSPEC_MODEL_REGISTRY` 可覆盖对应位置。标准 `pip install` 不执行联网的
post-install hook；使用这种安装方式后需另行执行 `vllm-hust-vspec-models`。
确实不希望安装器处理模型时可传 `--skip-model-setup`。

正式 wheel 安装与 Extension Manager 静态发现：

```bash
python -m pip install \
  "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"
python -m pip install /path/to/vllm_hust_vspec-0.13.2-py3-none-any.whl
vllm-hust-vspec-models
vllm-hust-ext extension inspect org.vllm-hust.vspec
```

安装包只使扩展可发现，不会自动启用补丁。Manager 读取包内静态 Manifest，发现
过程不会导入 `vllm_hust_vspec` 的运行时实现。

安装后可在加载模型前检查宿主 ABI：

```bash
vllm-hust-vspec-doctor --method draft
vllm-hust-vspec-doctor --method eagle --json
vllm-hust-vspec-doctor --method eagle --adaptive
```

检查结果包含 vLLM/vLLM-Ascend 版本、Git revision、dirty 状态和所需 API。
版本不等于测试基线时不会直接拒绝；缺少插件实际调用的方法时返回非零状态。

## Extension Manager 生命周期

当前 Manifest 的兼容范围精确限定为已经验证的 vLLM-HUST distribution
`0.17.2rc1.dev5871+g762f85b31.empty`。vLLM-Ascend 的投机解码接口没有独立
语义版本，因此在 Manifest 中明确标为未版本化，并由
`vllm-hust-vspec-doctor` 在运行前按真实 ABI 检查。已验证源码基线为：

- vLLM-HUST：`762f85b311fbab0bcf8921dd216f5093cd58b9b8`
- vLLM-Ascend-HUST：`4e57439e58ed3d78e675f9fd7b4614fb183c5394`
- Python：3.11
- 设备：Ascend NPU；模型族和运行约束见“支持范围”及各 TOML 配置

最新版软件栈已在独立环境 `/opt/vllm-hust-cann91` 中完成验证：CANN 9.1、
torch/torch-npu 2.13 和源码构建的 triton-ascend 3.6 均可用，宿主默认 CANN 9.0
未被覆盖。自定义算子导入、Triton NPU、插件完整测试和 Manager 生命周期已通过。
14B 单卡端到端复测目前因所有 NPU 仅余约 8 至 9.5 GiB 显存而等待资源；详情见
[`docs/latest_stack_validation_20260903.md`](docs/latest_stack_validation_20260903.md)。

Qwen2.5 EAGLE 所需的 `Qwen2ForCausalLMEagle` 已从该 core 版本移出内置模型
列表，因此由 vSpec 在启用时通过稳定的 `ModelRegistry.register_model` 插件接口
延迟注册；无需再修改 vLLM-HUST 的模型注册表。

完整启用流程：

```bash
vllm-hust-ext extension list --json
vllm-hust-ext extension validate org.vllm-hust.vspec
vllm-hust-ext extension inspect org.vllm-hust.vspec
vllm-hust-ext extension check org.vllm-hust.vspec
vllm-hust-ext extension plan org.vllm-hust.vspec
vllm-hust-ext extension render org.vllm-hust.vspec
vllm-hust-vspec-doctor --method eagle
vllm-hust-ext extension enable org.vllm-hust.vspec
vllm-hust-ext extension status org.vllm-hust.vspec

vllm-hust-ext run -- vllm-hust-vspec \
  --method eagle \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --gamma 2 \
  --graph-mode full \
  --async-scheduling \
  --prefix-caching
```

上述准入操作也可由统一入口执行：

```bash
./manage.sh list --json
./manage.sh validate
./manage.sh check
./manage.sh plan
./manage.sh render
./manage.sh enable
./manage.sh run --dry-run -- vllm-hust-vspec --config configs/qwen25-14b-eagle.toml
```

Manager 设置 `HUST_VSPEC_ENABLED=1`，具体 method、模型、gamma 和图模式仍由 vSpec
启动器提供。直接执行 `vllm-hust-vspec` 仍是受支持的显式启动方式，不受 Manager
保存的 enabled 状态控制。

停用、清理 Manager 意图和卸载：

```bash
./manage.sh disable
./manage.sh uninstall
# 等价快捷入口
./uninstall.sh
```

`uninstall` 依次执行 disable、forget 和 `pip uninstall vllm-hust-vspec`。它不终止
已经运行的 vLLM 进程；这些进程需重启后才会卸载插件。该流程不会删除 vLLM、
vLLM-Ascend、CANN、模型、KV 数据、NPU 驱动或共享服务。可使用
`./manage.sh --dry-run uninstall` 预览命令。

若目标 venv 使用 `--system-site-packages` 且父环境也安装了 vSpec，pip 只能删除
目标 venv 自己的 distribution。脚本会检测仍然可见的父环境副本并返回非零，不会
越过环境边界删除父环境包。正式部署建议使用不继承系统包的干净 venv。

精确版本升级与回退：

```bash
# 发布到 PyPI 后按版本升级
./manage.sh upgrade --version 0.13.2 --enable

# 本地 wheel 升级或回退
./manage.sh upgrade --wheel dist/vllm_hust_vspec-0.13.2-py3-none-any.whl
./manage.sh rollback --wheel dist/vllm_hust_vspec-0.12.1-py3-none-any.whl --enable
```

升级和回退都使用无缓存安装，并在完成后执行 Manifest validate 与宿主兼容性
check。它们不会停止现有 vLLM 进程，必须重启服务才能加载新版本。

## 构建与发布

版本唯一来源是 `src/vllm_hust_vspec/_version.py`；构建审计会强制检查 Python
发行版本、Manifest `extension_version` 和文件名一致。Manifest 位于独立、无运行时
副作用的 `vllm_hust_vspec.manifests` 包中，静态发现不会导入投机解码实现。

```bash
./release.sh source-check
./release.sh build
./release.sh check
```

`build` 优先执行 `uv build --no-sources`；本机没有 uv 时回退到
`python -m build`。构建前会删除 `dist/` 中旧 wheel/sdist，构建后要求目录中只有
当前版本的两个产物，并校验 Manifest、entry points、METADATA、RECORD、sdist 管理
脚本和 SHA256。

正式发布由 `v0.13.2` 形式的 Git tag 触发 `.github/workflows/release.yml`。手工发布
要求干净 Git 工作树、PyPI Token 和精确版本二次确认：

```bash
export UV_PUBLISH_TOKEN='<PyPI project token>'
export VSPEC_RELEASE_CONFIRM=0.13.2
./release.sh publish
unset UV_PUBLISH_TOKEN VSPEC_RELEASE_CONFIRM
```

Token 只从环境或 CI Secret 读取，不写入源码、配置和日志。当前目录不是 Git checkout
时，`publish` 会拒绝执行；可以正常使用 `source-check`、`build` 和 `check`。

## 启动

```bash
# 串行 Draft，默认 B128、gamma 上限 4、FULL_DECODE_ONLY、Online Adaptive
./run.sh draft

# 严格 EAGLE，默认 B128、gamma 上限 4、FULL、Online Adaptive
./run.sh eagle

# 工程回归协议：ARC-Easy、B16、FP16、FULL_DECODE_ONLY、端口 18180
./run.sh draft-arc-easy
./run.sh eagle-arc-easy

# 明确选择近似验收配置
./run.sh eagle-relaxed
```

预设后可继续传入参数覆盖 TOML：

```bash
./run.sh eagle --device 2 --port 8100 --gamma 3 --max-num-seqs 64
./run.sh draft --graph-mode full-decode-only --capture-policy auto

# 恢复固定 gamma
./run.sh draft --no-adaptive-speculation
```

无需离线 profile 的 vSpec Online Adaptive 已默认启用。以下命令显式列出完整参数，
便于覆盖默认值：

```bash
./run.sh draft \
  --gamma 4 \
  --dtype bfloat16 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 33280 \
  --max-model-len 1024 \
  --prefix-caching \
  --graph-mode full \
  --adaptive-speculation \
  --adaptive-policy online \
  --adaptive-min-gamma 1 \
  --adaptive-control-interval 4 \
  --adaptive-online-warmup-samples 1 \
  --adaptive-online-exploration 0.01 \
  --adaptive-hysteresis 0.05 \
  --adaptive-full-graph \
  --adaptive-async
```

Qwen2.5 14B/0.5B、B128 的已验证性能预设可直接启动：

```bash
./run.sh draft-adaptive
```

该预设使用 `--adaptive-refill-batch 8` 和动态 gamma 的精确图桶，并显式设置
`--confidence-accept-margin 5.25`。margin 会放宽 greedy 验收，不保证逐 token 与
Target-only 完全一致；不接受该质量折中的场景应删除此参数，继续使用严格验收。

这里的 `--gamma 4` 是运行时搜索、静态 buffer 和 Graph capture 的上界，不会
把每轮投机长度固定为 4。`online` 模式不接收 `--adaptive-profile`。
需要在该预算内逐请求截断当前轮低置信后缀时，额外启用
`--adaptive-entropy-stop`；它默认关闭，且不限制下一轮 UCB 选择。

### ARC-Easy 在线回归

`--protocol arc-easy` 与工程回归基线使用相同的 Target、FP16、B16、端口、长度、
缓存和 `FULL_DECODE_ONLY` 设置。动态 gamma 默认在 `1..4` 内在线选择，无需传入
`--gamma`。安装在文档容器的 Python 环境后可直接运行：

```bash
# Draft 服务
/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method draft \
  --draft-model /model/Qwen2.5-0.5B-Instruct

# 对应压测
/usr/local/python3.11.14/bin/vllm-hust-vspec-bench --method draft
```

严格 EAGLE 使用相同协议，只替换投机方法和 Drafter：

```bash
/usr/local/python3.11.14/bin/vllm-hust-vspec \
  --protocol arc-easy \
  --method eagle \
  --draft-model /model/Eagle-Qwen2.5-14B-Instruct

/usr/local/python3.11.14/bin/vllm-hust-vspec-bench --method eagle
```

压测入口默认读取 `/run_dir/materialized.jsonl`，固定 200 条 prompt、输出 256 token、
无限请求速率、`temperature=0`，结果分别写入
`/run_dir/benchmark_results/ARC-Easy-{draft,eagle}-vspec/ARC-Easy.json`。基线仍使用
准备文档原命令；容器启动和数据物化步骤不变。动态 gamma 会自动补齐验证阶段所需
Graph bucket，因此 capture sizes 会大于 Target-only 的 `1 2 4 8 16`。可先用
`--dry-run` 查看最终 `vllm serve` 或 `vllm bench serve` 命令。

启用离线 profile 策略：

```bash
./run.sh draft \
  --gamma 4 \
  --dtype bfloat16 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 33280 \
  --max-model-len 1024 \
  --prefix-caching \
  --graph-mode full \
  --adaptive-speculation \
  --adaptive-profile profiles/qwen25_draft_gsm8k_graph_b128_g4.json \
  --adaptive-min-gamma 3 \
  --adaptive-min-observations 1 \
  --adaptive-full-graph \
  --adaptive-async
```

EAGLE 的已验证配置：

```bash
./run.sh eagle \
  --gamma 2 \
  --dtype bfloat16 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --max-model-len 1024 \
  --prefix-caching \
  --graph-mode full \
  --adaptive-speculation \
  --adaptive-profile profiles/qwen25_eagle_gsm8k_graph_b128.json \
  --adaptive-min-gamma 1 \
  --adaptive-min-observations 1 \
  --adaptive-full-graph \
  --adaptive-async \
  --eagle-spec-metadata-cache \
  --graph-event-ordering
```

Adaptive 默认使用同步调度和 `PIECEWISE`，作为兼容性优先路径。显式启用
`--adaptive-full-graph --adaptive-async` 后，会为每个候选 gamma 隔离、捕获并
回放 FULL Draft Graph，同时维护异步下一帧状态。Profile 必须用对应模型和运行
配置实测校准。

仓库中的 `qwen25_draft_gsm8k_graph_b128_g4.json` 和
`qwen25_eagle_gsm8k_graph_b128.json` 仅对报告中的 BF16、B128、prefix caching、
`max_model_len=1024` 配置完成了校准；改变 dtype、batch、模型或主要输入分布后
应重新测量，不能把示例 profile 当成通用默认值。

从 forward 延迟样本拟合非负线性 profile：

```bash
vllm-hust-vspec-fit-profile samples.json profile.json
```

查看环境变量和最终 `vllm serve` 命令，不加载模型：

```bash
./run.sh eagle --dry-run
```

使用自定义配置：

```bash
VSPEC_CONFIG=/path/to/config.toml ./run.sh eagle
```

新接口使用 `VSPEC_CONFIG` 和 `HUST_VSPEC_*`。为兼容已有实验脚本，旧
`SPECSLO_CONFIG`、`HUST_SPECSLO_*`、`SPECASCEND_CONFIG` 和
`HUST_SPECASCEND_*` 仍可读取，但新变量优先。

额外原生 vLLM 参数放在 `--` 后：

```bash
./run.sh eagle --port 8100 -- --generation-config vllm
```

## 通用参数

- `--method draft|eagle`
- `--target-model` / `--draft-model`
- `--gamma`
- `--max-num-seqs` / `--max-num-batched-tokens` / `--max-model-len`
- `--graph-mode eager|piecewise|full-decode-only|full`
- `--generation-config vllm|auto`（默认 `vllm`）
- `--capture-policy auto|exact|power2|steady`
- `--adaptive-refill-batch`（`0` 为方法默认值：Draft 8，EAGLE 4）
- `--confidence-accept-margin`（可选近似验收；默认关闭）
- `--confidence-accept-from-position` / `--confidence-accept-after-tokens`
- `--confidence-protected-token-ids`
- `--async-scheduling` / `--no-async-scheduling`
- `--prefix-caching` / `--no-prefix-caching`
- `--chunked-prefill` / `--no-chunked-prefill`
- `--device` / `--port` / `--gpu-memory-utilization`
- `--adaptive-speculation` / `--adaptive-policy profile|online`
- `--adaptive-profile PATH`（仅 `profile` 策略）
- `--adaptive-ewma-weight` / `--adaptive-hysteresis`
- `--adaptive-min-gamma` / `--adaptive-min-observations`
- `--adaptive-control-interval` / `--adaptive-max-gamma-step`
- `--adaptive-latency-calibration` / `--no-adaptive-latency-calibration`
- `--adaptive-latency-ewma-weight`
- `--adaptive-gamma0-mode sticky|sync`
- `--adaptive-online-window` / `--adaptive-online-exploration`
- `--adaptive-online-warmup-samples`
- `--adaptive-online-warmup-return best|incumbent`
- `--adaptive-entropy-stop` / `--adaptive-entropy-topk`
- `--adaptive-entropy-threshold` / `--adaptive-entropy-scale`
- `--adaptive-trace`
- `--adaptive-full-graph` / `--adaptive-async`

`--adaptive-min-gamma 0` 允许控制器关闭投机。默认 `sticky` 是真正的 target-only：
跳过 Draft forward/prefill，直到禁用期间的请求 cohort 全部结束后才对新 cohort
恢复投机。`sync` 会执行一个不提交 token 的 gamma=1 shadow Draft step，使同一请求
能从 gamma=0 恢复，但 gamma=0 不再是零 Draft 计算。此时推荐
`--graph-mode full-decode-only`，可为 Target width 1 保留独立 decode 图；组合
`FULL` 会对 target-only step 安全回退 eager。

动态 gamma 在 async 模式下允许相邻帧使用不同验证宽度。切换帧若与当前 Graph
capture bucket 不整除，会只对该帧回退 eager。`FULL_DECODE_ONLY` 的 Target 图参数
按 query width 隔离；组合 `FULL` 必须共享 Target 图参数，Draft 图仍按 gamma 隔离。

在线时延校准默认启用。`profile` 策略用它修正离线模型的 gamma 排名；`online`
策略直接将已提交输出 token 数除以 scheduler 消费输出时的非重叠完成间隔作为
实测 reward。启用当前轮置信停止后，`gamma=1/2` 复用原始图，不捕获额外 entropy
计算；Online-UCB 的完成反馈仍覆盖所有 gamma。图模式不要设置
`TORCHDYNAMO_DISABLE=1`，该变量会使
`--compilation-config` 的图编译失败。

## Draft 功能

- merged FULL Draft Graph
- compact PIECEWISE Draft Graph
- Graph request/token padding
- `query_start_loc` 补齐
- Draft KV slot mapping 修正
- Qwen2.5 共享 token ID、不同 vocab padding 的快速路径
- FULL/PIECEWISE Draft 路径按 batch 选择

主要参数：

- `--shared-tokenizer-padding` / `--no-shared-tokenizer-padding`
- `--merged-full` / `--no-merged-full`
- `--merged-full-max-batch N`

共享 tokenizer 快速路径只适用于已验证的 Qwen2.5-0.5B + 14B 模型对。
其他 Draft/Target 组合应使用 `--no-shared-tokenizer-padding`。

## EAGLE 功能

- Qwen2 EAGLE 缺失 QKV bias 的确定性初始化
- 线性 EAGLE 验收热路径，移除逐请求 `.item()` D2H 同步
- 严格 greedy top-1 验收
- EAGLE FULL Graph capture 参数生成
- 可选 Draft/Target active vocabulary
- 可选 Draft LM-head W8A16/W8A8
- 可选 Graph event ordering、metadata cache、uniform-state kernel
- 可选固定宽度 EAGLE tree 实验路径
- 可选 top-K relaxed acceptance 和严格前缀

主要参数：

- `--eagle-draft-active-vocab-size N`
- `--eagle-draft-active-vocab-ids PATH`
- `--eagle-target-active-vocab-ids PATH`
- `--eagle-draft-lm-head-quantization none|w8a16|w8a8`
- `--eagle-relaxed-accept-topk K`
- `--eagle-relaxed-accept-after-tokens N`
- `--eagle-spec-metadata-cache`
- `--eagle-uniform-state-kernel`
- `--graph-event-ordering`
- `--eagle-tree-width N`
- `--eagle-zero-draft-kv-first-step`

诊断与 active-vocab 数据采集参数：

- `--eagle-draft-trace`
- `--eagle-draft-io-trace-dir PATH`
- `--eagle-target-hidden-trace-dir PATH`
- `--eagle-target-argmax-trace-path PATH`

诊断参数会触发 NPU 同步或写盘，只用于正确性排查和 active-vocab 构建，
不应在正式吞吐测试中启用。

`--eagle-relaxed-accept-topk` 大于 1 时会改变生成结果，只应用于明确接受
性能/质量折中的场景。EAGLE tree 当前仍是实验功能，线性 EAGLE 是稳定默认。

插件默认使用 `--generation-config vllm`，避免模型自带的 repetition penalty、
top-k 等默认值破坏 active-vocab 的纯 greedy 前提。选择 `auto` 后，调用方必须
为 active-vocab 请求显式使用 `temperature=0`、`repetition_penalty=1` 且不启用
logprobs、约束解码或其他 logits processor。

## 代码所有权

以下实现已经物理迁入本插件，不依赖工作区中的同名未提交实现：

- Draft merged FULL/compact PIECEWISE、padding 和 KV slot 修正
- Qwen2 EAGLE QKV bias 初始化
- 线性 EAGLE 无 D2H 验收路径
- FP16 Draft/Target active-vocab LM-head 和完整 token-ID 映射
- Draft active-vocab LM-head W8A16/W8A8 分块量化
- strict top-1、relaxed top-K 和严格前缀验收
- ACL Graph device-event ordering
- Draft torch-compile 控制和 Draft/Target 共享模块隔离
- uniform async EAGLE speculative metadata cache
- Target hidden state 保留
- 首次 Draft KV 清零、Draft token/Target hidden/Target argmax trace
- vSpec Adaptive Goodput 控制器、在线验收反馈和运行时动态 Draft 长度

以下深层能力仍由配套 `vllm-ascend-hust` ABI 提供：

- uniform-state kernel
- 固定宽度 EAGLE tree、Target-width 和 tree Graph commit
- Draft 内部 IO trace

插件会在这些参数被显式启用时检查 host 源码是否包含对应实现。缺失时立即
报错，不会静默忽略开关。

## 请求生成参数

`max_tokens`、`temperature`、`top_p` 等属于请求参数：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-14b-eagle-fp16",
    "messages": [{"role": "user", "content": "Solve: 17 * 23"}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

## 加载与依赖边界

插件只有在 `HUST_VSPEC_ENABLED=1` 时应用补丁；普通 vLLM、Baseline
和未通过本目录启动的任务不受影响。

插件不复制完整的 scheduler、attention 和模型执行器，而是在公开类边界注入
本目录中的 Draft/EAGLE 实现。它仍依赖 `/root/data/vllm-hust-latest` 与
`/root/data/vllm-ascend-hust-latest` 提供基础投机解码 ABI；具体所有权见上一节。
