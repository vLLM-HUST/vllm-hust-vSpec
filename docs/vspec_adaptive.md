# vSpec Adaptive

vSpec Adaptive 是插件内的闭环投机长度控制器。它以 `--gamma` 为运行时上界，
根据当前 batch、在线验收结果和已完成 step 的耗时，在每个 decode step 选择一个
batch-uniform `gamma`。这个上界同时决定静态 buffer 和 Graph capture 范围，因此
启动时仍需指定，但运行时不固定使用该值。

串行 Draft、EAGLE 和 EAGLE3 默认启用 `online` 策略，默认 gamma 上限为 4，候选范围
为 `1..4`，并默认保留 async scheduling
和 FULL/FULL_DECODE_ONLY Graph。固定 gamma 实验必须显式传入
`--no-adaptive-speculation`。DFlash 尚不支持 online 策略，因此未显式指定 profile
时会保持 Adaptive 关闭。

## 控制策略

`--adaptive-policy profile` 使用离线 Draft/Target forward 延迟模型作为冷启动，
再用在线逐位置验收率和已完成 step 的实测时延修正 Goodput 排名。Profile 还可
提供 `batch_gamma_policy` 作为活动 batch bucket 的初始锚点；在线预测超过锚点和
hysteresis 后，可用 `online_policy_override` 覆盖它。

`--adaptive-policy online` 不读取 profile，也不需要部署前硬件实验。它按二次幂
划分活动 batch bucket，以 gamma 为 UCB arm，并直接计算：

```text
reward(gamma, bucket) = committed_output_tokens / completion_interval
```

普通候选范围冷启动时从最大可执行 gamma 开始，按相邻档位逐级向下探索；包含
`gamma=0` 且跨度至少为 4 的宽范围先在候选中点完成安全 burn-in，再从相邻档位
逐级探索。默认 burn-in 为 `8 * adaptive_online_window` 个稳定观测，避免短任务为
完整 arm 扫描付出过高启动成本。每个 arm 调度一个驻留窗口，async 模式多填充
一步以跨过 gamma 切换帧。反馈未返回时控制器在候选区间中点等待。UCB 使用滑动
reward、探索 bonus 和对称 hysteresis 排名；候选收益接近时优先选择靠近中点的
档位，gamma 每次最多改变 `--adaptive-max-gamma-step`。新 batch bucket 从相邻
bucket 继承在线先验和已训练 gamma，不会在短尾段重新执行整轮探索。

可选 `--adaptive-entropy-stop` 是独立的当前轮逐请求停止层。它在 Draft logits
上计算 top-k entropy，满足
`entropy * scale > threshold^2` 时保留当前位置 token，并把本轮后缀标为无效；
它不限制下一轮 UCB gamma。FULL Graph 下只在 gamma3 及以上捕获该计算，gamma1/2
复用原始图。静态图已经执行的 Draft forward 无法撤销，因此该开关默认关闭。

## Profile 控制模型

Target 和 Draft 各使用一个离线拟合的线性 forward 延迟模型：

```text
T_forward = alpha * context_tokens + gamma * batched_tokens + delta
```

对候选投机长度 `k`，控制器预测：

```text
T_step(k) = sum(T_draft(i), i=0..k-1) + T_target(k + 1) + overhead(k)
E[tokens(k)] = 1 + p1 + p1*p2 + ... + p1*p2*...*pk
Goodput(k) = batch_size * E[tokens(k)] / T_step(k)
```

上式适用于串行 Draft/EAGLE。DFlash profile 设置 `draft_parallel: true`，其 Draft
成本改为一次并行 block forward：`T_draft(context, batch * (k + 1))`，不会错误地
按 k 次 forward 累加。

在线 AC 不是简单的 `accepted/proposed`。控制器分别维护每个 Draft 位置的条件
验收率 EWMA；每个请求只把成功前缀和首次拒绝计入对应位置的 Bernoulli 试验，
避免把首次拒绝后的未验证 token 当作失败样本。汇总 AC 仍保留用于 trace。

在线时延校准默认开启。稳定 decode step 完成后，控制器计算“实测时延 / profile
预测时延”的逐 gamma EWMA；尚未观测的 gamma 使用全局修正因子。这样离线 profile
负责冷启动，运行中的 NPU、Graph 和 batch 实测结果负责修正 Goodput 排名。

## Profile 格式

Profile 必须来自同一 Target/Draft 模型、dtype、图模式和主要 batch/context
范围的离线测量。以下 JSON 只展示字段，不是可直接用于性能实验的校准结果：

```json
{
  "schema_version": 1,
  "max_speculative_tokens": 4,
  "initial_gamma": 4,
  "default_acceptance_rate": 0.75,
  "draft_parallel": false,
  "target": {
    "alpha_ms_per_context_token": 0.001,
    "gamma_ms_per_batched_token": 0.01,
    "delta_ms": 1.0
  },
  "draft": {
    "alpha_ms_per_context_token": 0.0001,
    "gamma_ms_per_batched_token": 0.001,
    "delta_ms": 0.1
  },
  "per_gamma_overhead_ms": {
    "1": 0.0,
    "2": 0.0,
    "3": 0.0,
    "4": 0.0
  },
  "batch_gamma_policy": {
    "64": 4,
    "128": 3
  }
}
```

`per_gamma_overhead_ms` 是可选字段，用来吸收 rejection sampler、padding、
host 调度和特定图 bucket 产生的固定开销。所有时间单位均为毫秒。
`batch_gamma_policy` 也是可选字段；key 是活动 batch 上界，value 是该 bucket
的校准 gamma，允许为 `0`。超过最大 key 时使用最后一个 bucket。该字段适合保存端到端
A/B 得到的非线性结论，拟合工具会原样校验并保留它。

至少采集覆盖目标工作负载的三组线性独立数据点；实际建议覆盖
`B8/B32/B64/B128`、短/中/长上下文和所有候选 `gamma`，再用最小二乘拟合
Target、Draft 的三个系数。Profile 的 `max_speculative_tokens` 必须大于等于
启动参数 `--gamma`。

插件提供非负最小二乘拟合工具。输入格式如下：

```json
{
  "max_speculative_tokens": 4,
  "initial_gamma": 4,
  "default_acceptance_rate": 0.75,
  "batch_gamma_policy": {"64": 4, "128": 3},
  "target_samples": [
    {"context_tokens": 1024, "batched_tokens": 8, "latency_ms": 12.3}
  ],
  "draft_samples": [
    {"context_tokens": 1024, "batched_tokens": 8, "latency_ms": 2.1}
  ]
}
```

每个数组至少需要三条线性独立样本，以上单条仅说明字段。生成 profile：

```bash
vllm-hust-vspec-fit-profile samples.json profile.json
```

也可以直接运行自动采集 suite。它会逐个执行固定 gamma 基准、开启宿主分阶段
计时、保留原始 JSON/log，并拟合 forward 模型和按吞吐选择 batch policy：

```bash
vllm-hust-vspec-profile \
  --method draft \
  --execution-mode full-decode-only \
  --batches 8,32,64,128 \
  --gammas 0,1,2,3,4 \
  --max-tokens-grid 128,512 \
  --dataset gsm8k \
  --manifest /path/to/qwen25_14b.json \
  --num-prompts 200 \
  --max-num-batched-tokens 33280 \
  --device 0 \
  --async-scheduling \
  --output-dir benchmark_results/profile_qwen25_draft \
  --output-profile profiles/qwen25_draft_auto.json
```

`gamma=0` 会以无 speculative config 的真实 Target baseline 运行并参与 policy
选择；正 gamma 运行同时提供 Draft 拟合样本。`measurements.json` 保存每个样本的
输入、累计 component stats 和源文件路径，最终 profile 可追溯到原始测量。
suite 会为所有 gamma 固定同一个 `max_num_batched_tokens`；未显式指定时按最大
batch 和最大 gamma 计算一次，不会逐测量改变 KV/cache 配置。
suite 默认启用 async scheduling；用 `--no-async-scheduling` 可为同步部署生成
匹配 profile。

DFlash 使用 vLLM-Ascend 原生并行 proposer，模型路径必须显式提供：

```bash
vllm-hust-vspec-profile \
  --method dflash \
  --target-model /path/to/target \
  --draft-model /path/to/dflash \
  --manifest-family qwen3_8b \
  --execution-mode full \
  --batches 8,32,64,128 \
  --gammas 0,1,2,3,4 \
  --max-tokens-grid 128,512 \
  --dataset gsm8k \
  --manifest /path/to/manifest.json \
  --output-dir benchmark_results/profile_dflash \
  --output-profile profiles/dflash.json
```

生成的 profile 会自动写入 `draft_parallel: true`。
启动时会校验该字段：DFlash 拒绝串行 profile，Draft/EAGLE 拒绝并行 profile，
避免使用错误的时延公式静默做出 gamma 决策。

## 启动

无需离线 profile 的在线模式：

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
  --adaptive-online-window 32 \
  --adaptive-control-interval 4 \
  --adaptive-online-warmup-samples 1 \
  --adaptive-online-exploration 0.01 \
  --adaptive-hysteresis 0.05 \
  --adaptive-full-graph \
  --adaptive-async
```

当前轮置信停止需要时额外添加：

```bash
--adaptive-entropy-stop \
--adaptive-entropy-topk 2 \
--adaptive-entropy-threshold 0.3 \
--adaptive-entropy-scale 0.15
```

离线 profile 模式：

```bash
./run.sh draft \
  --gamma 4 \
  --dtype bfloat16 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 33280 \
  --max-model-len 1024 \
  --prefix-caching \
  --graph-mode full-decode-only \
  --adaptive-speculation \
  --adaptive-profile profiles/qwen25_draft_gsm8k_graph_b128_g4.json \
  --adaptive-ewma-weight 0.1 \
  --adaptive-hysteresis 0.03 \
  --adaptive-min-gamma 0 \
  --adaptive-min-observations 1 \
  --adaptive-max-gamma-step 1 \
  --adaptive-latency-calibration \
  --adaptive-latency-ewma-weight 0.2 \
  --adaptive-gamma0-mode sticky \
  --adaptive-full-graph \
  --adaptive-async
```

也可用于线性 EAGLE/EAGLE3。启用后 launcher 会执行以下约束：

- 默认情况下 Eager 保持 Eager，Graph 转成 `PIECEWISE`，并关闭 async。
- `--adaptive-full-graph` 保留 FULL/FULL_DECODE_ONLY，为每个 gamma 捕获独立
  Draft Graph 和 graph-parameter table。
- `--adaptive-async` 保留 async scheduling，并在 gamma 改变后修复下一帧
  placeholder 和 D2H 宽度。
- Draft 的 merged FULL Graph 只在显式 FULL opt-in 时保留。
- EAGLE tree 和 Target-width 模式拒绝启动。
- Target decode 图覆盖 `1..max_gamma+1` 的全部 query width；正 gamma 的 Draft
  图按候选长度独立捕获，`gamma=0` 不捕获 Draft 图。`FULL_DECODE_ONLY` 的
  Target graph-parameter table 按 query width 隔离，组合 `FULL` 则使用共享 table，
  否则 Ascend 的共享 mixed event family 会在重复回放时死锁。
- async gamma 切换帧若落入与旧 query width 不整除的 capture bucket，会转为
  非 uniform/eager 分发；下一稳定帧恢复对应宽度的 FULL decode 图。
- 候选会按 `max_num_batched_tokens` 检查 `batch * (gamma + 1)`，不可执行的
  gamma 不参与 Goodput 比较。

要允许控制器关闭投机，设置 `--adaptive-min-gamma 0`。默认
`--adaptive-gamma0-mode sticky` 会跳过 Draft forward/prefill，并将当时所有活跃
请求标记为 target-only；这些请求以及禁用期间到达的新请求全部结束后，后续请求
才会重新启用投机。这个 sticky cohort 规则避免用已经失步的 Draft KV 恢复投机。

`--adaptive-gamma0-mode sync` 允许同一请求恢复：gamma=0 期间执行一个不提交候选
token 的 gamma=1 shadow Draft step，使 Draft KV 保持同步。它已验证 `0 -> 1 -> 2`
切换和输出一致性，但不是零 Draft 计算，因此只应在周期性恢复的收益高于 shadow
开销时启用。

当候选包含 `gamma=0` 时，推荐 `--graph-mode full-decode-only`。该模式的
uniform decode 图键包含 query width，Target-only 的 width 1 可以与投机验证宽度
并存。组合 `FULL` 只有一套 mixed descriptor family，width 1 可能与按最大验证宽度
捕获的静态 FIA metadata 冲突；插件会把这种 target-only step 明确回退到 eager，
保证输出正确。正 gamma 不受该回退影响。

`--adaptive-trace` 会记录 batch、context、逐位置条件 AC、前后 `gamma`、预测
Goodput 和决策原因，只用于校准和诊断。

对应环境变量为：

- `HUST_VSPEC_ADAPTIVE_POLICY`
- `HUST_VSPEC_ADAPTIVE_LATENCY_CALIBRATION`
- `HUST_VSPEC_ADAPTIVE_LATENCY_EWMA_WEIGHT`
- `HUST_VSPEC_ADAPTIVE_GAMMA0_MODE`
- `HUST_VSPEC_ADAPTIVE_ONLINE_WINDOW`
- `HUST_VSPEC_ADAPTIVE_ONLINE_EXPLORATION`
- `HUST_VSPEC_ADAPTIVE_ONLINE_WARMUP_SAMPLES`
- `HUST_VSPEC_ADAPTIVE_ENTROPY_STOP`
- `HUST_VSPEC_ADAPTIVE_ENTROPY_TOPK`
- `HUST_VSPEC_ADAPTIVE_ENTROPY_THRESHOLD`
- `HUST_VSPEC_ADAPTIVE_ENTROPY_SCALE`

图模式启动时不要设置 `TORCHDYNAMO_DISABLE=1`。它会禁用 AOT compile，与
`FULL`/`FULL_DECODE_ONLY` 的 `compilation_config` 冲突。

动态 EAGLE 会先隔离 `VLLM_ASCEND_EAGLE_UNIFORM_STATE_KERNEL`，因为该上游
快路径默认按启动时固定的 `num_spec_tokens` 修正异步 optimistic state，直接跨越
`gamma=4 -> 3` 会造成 Draft KV/hidden state 失步。插件在 Graph 初始化后把
Target 状态宽度锚定到候选中点；只有当前 Target query width、运行时 gamma 与
状态宽度一致的稳定帧才临时启用原生快路径。正 gamma 降档过渡帧按旧宽度执行
proposal，再把候选截断到新宽度。固定 gamma 运行不受这些动态保护逻辑影响。

## 当前边界

当前候选范围是 `min_gamma..max_gamma`，支持串行 Draft、线性 EAGLE/EAGLE3、
原生 DFlash 并行 proposer 的动态宽度钩子、FULL/FULL_DECODE_ONLY 多 gamma Graph、
异步调度、两种 gamma=0 策略、Draft prefill gating、profile 校准、无 profile 在线
Goodput 搜索、当前轮逐请求置信停止和 batch-token budget 过滤。

DFlash 代码已经接到 vLLM-Ascend 原生 proposer，但本机没有匹配的 DFlash checkpoint，
目前只有 ABI/单测覆盖，尚无 NPU 端到端结论。未实现的边界是 PLD 独立
proposal/verification 长度、连续请求下同时做到零 Draft 计算和同请求周期性恢复的
per-request 混合状态。离线 profile 自动采集 suite 仍会为每个测量重新启动模型；
不希望承担这项部署成本时使用 `--adaptive-policy online`。
`--adaptive-trace` 会产生逐 step 日志，只用于校准，不能用于正式吞吐测试。
