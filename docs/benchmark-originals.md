# Benchmark originals

[Remaining regression manifest](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/vspec-regression/manifest.json)
preserves all 316 remaining tracked `benchmark_results/` files. The downloader combines these with
the previously migrated GSM8K originals at their historical relative paths. Diagnostic/eager runs,
failures and original model identities remain separate from qualified results. Commands in old
reports describe historical environments; migration does not recreate missing local datasets.

This bounded collection is preserved by the benchmark repository. Original versions, failures,
negative results and diagnostic scope remain unchanged; migration is not new runtime acceptance.
Other historical collections are not declared migrated by this change.

- [Original manifest 1](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/vspec-browser-raw-20261022/manifest.json)
- [Original manifest 2](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9b81690f767aca56524ec6651535a9b8d484a4c3/archive/mod-git-20261022/website-source-vllm-hust-vSpec/manifest.json)

Download and verify each original into an empty directory outside the source checkout:

```bash
python scripts/fetch_benchmark_originals.py /absolute/external/originals
```

The downloader checks the pinned manifest and each file's length and SHA256, rejects unsafe paths
and never executes archived code. CI repeats the readback and rejects the migrated paths returning
to Git. Fixed benchmark originals and unchanged source Git history preserve recoverability.

[Additional legacy numerical narratives](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2a97ad593c4fc3a6df051ec6dc3b78675a7a6225/archive/mod-targeted-20261023/vspec-legacy-narratives/manifest.json) preserves three complete reports and LICENSE.
Current report entry points link each referenced raw result to the existing fixed benchmark capsules;
brace expressions are expanded to verified members. Historical defaults, quality differences and
negative results remain in the originals and are not promoted to current acceptance.

[vspec-adaptive-narratives](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/5eeccc2d69b2c4a634e273a9f926584c8d252cf7/archive/mod-targeted-20261023/vspec-adaptive-narratives/manifest.json) — Adaptive NPU and dynamic-gamma historical reports plus LICENSE; direct JSON/log links are checksum-bound. Quality regressions and diagnostic limits remain unchanged.

[B16 / EAGLE / Graph-Eager / Frontier originals](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/9e01d8b6870dcbefd474b1102ef7f01e2a662ec9/archive/mod-targeted-20261023/vspec-b16-frontier/manifest.json) preserve six complete historical documents and LICENSE. Raw links reuse existing capsules; the reference audit names its exact manifest scope and distinguishes unavailable inputs from future output examples. Historical plans do not authorize new runs.

[AgentX / SPEED-Bench / School 0.28 originals](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/51604897aabb9a856b311fa58fc6749938d0378e/archive/mod-targeted-20261023/vspec-agentx-school/manifest.json) preserve three complete documents and LICENSE. Ten existing raw members are linked directly; twelve specifically named logs/results are unavailable in the pinned capsule scope. School summary remains in its separate canonical benchmark collection.
