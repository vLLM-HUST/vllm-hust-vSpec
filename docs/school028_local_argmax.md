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
logits processor, model interfaces and Qwen3.5 MTP source hashes. The proposer pin
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
proposer's merged `_runnable` graph, rather than mistaking the draft model object
for the captured callable. Performance admission additionally requires matching
source/import identity, all 200 original requests, Microflow and KV control
receipts, no foreign NPU load, paired cache initialization, clean drain and device
release. Startup failures are diagnostic records, never performance points.

## Status

CPU and Manager admission have been exercised. NPU correctness, graph execution
and same-workload performance qualification must complete before this adapter is
selected as a published B1. Existing Microflow measurements are not inherited as
evidence for this additional optimization.
