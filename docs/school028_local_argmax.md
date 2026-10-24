# School 0.28 MTP2 Local Argmax

This default-off experimental adapter isolates the existing
`_sample_mtp_local_argmax` mechanism. It does not activate the full vSpec patchset,
adaptive gamma, cohort refill, state remapping, or a different drafter.

## Configuration

- Qwen3.5-35B-A3B BF16, TP2, native MTP2, greedy sampling.
- Current source-pinned BetterScale E24/R36 + KV recompute + FA Microflow B1.
- Original MMLU-Pro 200-request burst, infinite arrival rate, out256.
- APC, async, FULL graphs and the existing state/host budgets remain unchanged.

The additional bundle is `org.vllm-hust.vspec-school028-argmax`. Extension Manager
must also admit `org.vllm-hust.prefill-microflow-school028-fa`; its full source
preflight is reused. The school adapter additionally checks the exact proposer,
logits processor, model interfaces, Qwen3.5 MTP and FIA publisher source hashes. The proposer pin
is the prepared BetterScale donor, not the unmodified installed base package.
No installed host file is edited.

Select `HUST_VSPEC_SCHOOL_MODE=off` or `on` and a writable
`HUST_VSPEC_SCHOOL_EVENT_DIR` in an isolated process. The Manager activation adds
`vspec_school028_argmax` and `HUST_VSPEC_SCHOOL028_ARGMAX=1`; direct activation of
the full `vspec` entry point is rejected. Restarting without the bundle restores
native behavior.

## Qualification

Both arms check real BF16 head weights on both TP ranks, comparing full-logit
sampling with local argmax for 1, 24, 36 and 48 rows. Zero-valued hidden states
exercise tied logits; deterministic nonzero hidden states exercise ordinary
selection. Any token mismatch aborts model loading. These checks do not replace
whole-task accuracy evaluation.

The observer records selector execution by shape and actual replay of the
native merged graph inside the pinned FIA publisher's `inner` callable, rather
than mistaking the draft model or outer publisher for the ACL graph.
Performance admission additionally requires matching
source/import identity, all 200 original requests, Microflow and KV control
receipts, no foreign NPU load, paired cache initialization, clean drain and device
release. Startup failures are diagnostic records, never performance points.

## Historical result and evidence

The complete historical result, negative/near-zero comparison, qualification
limits, failed starts and test receipts are preserved in the
[frozen benchmark original](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/51604897aabb9a856b311fa58fc6749938d0378e/archive/mod-targeted-20261023/vspec-agentx-school/snapshot/docs/school028_local_argmax.md). No historical number is promoted to a new B1.

Read the [raw-run collection](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/9ab6413d1ea47587953f037d42fea568f4a5a201/reports/vllm-hust-midterm-20261019/vspec-school028)
and its [machine-readable summary](https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/9ab6413d1ea47587953f037d42fea568f4a5a201/reports/vllm-hust-midterm-20261019/vspec-school028/summary.json). Task accuracy was not rescored;
bounded head equivalence does not establish whole-task quality equivalence.
Startup failures remain separate from performance points. This migration does
not clear the historical global-runtime ABI blocker or authorize a hardware rerun.

下载与 SHA 核验见[原件说明](benchmark-originals.md)；精确绑定和缺失范围见 [BENCHMARK_ORIGINALS.json](../BENCHMARK_ORIGINALS.json)。
