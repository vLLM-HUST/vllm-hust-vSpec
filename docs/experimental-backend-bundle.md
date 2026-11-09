# Experimental Qwen3.5 backend source bundle

This development bundle stages source code from the experimental Qwen3.5
execution path. It is not a complete installer and is not an NPU validation or
an assertion that historical benchmark results can be reproduced.

The tracked backend patch and additional Python sources apply to the
`vLLM-HUST/vllm-ascend-hust` source tree at commit
`4e57439e58ed3d78e675f9fd7b4614fb183c5394` (Apache-2.0). The repository
license also covers the changes carried here; original upstream notices must
remain intact. The plugin wheel carries the patch and additional source files.

`vllm-hust-vspec-prepare-backend --source CHECKOUT --destination NEW_DIR`
creates a separate source checkout at that pinned commit, verifies that the
patch applies, and adds the extra files. The command refuses to overwrite an
existing destination or place it inside the source checkout. It does not build
or install software, modify the original checkout, provision system drivers or
the CANN SDK, or bypass any correctness checks.

The wheel also includes selected standalone Triton kernel sources under
`vllm_hust_vspec/_vendor/operators`. These sources are not automatically
selected or registered by this staging operation. The corresponding
experimental runtime-hook source snapshots are included under
`vllm_hust_vspec/_vendor/experimental_hooks`; they are not automatically
enabled and retain fail-closed validation requirements. The historical
experimental worker dependencies are included under `_vendor/runtime`; they
are not enabled by default. Their component proofs are checked before worker
hook installation, but the migrated worker has not been integration-tested or
established as equivalent to the historical serving setup. Experimental
Ascend convolution and recurrent operator sources and a build entry point are included:

`vllm-hust-vspec-build-operator --cann-root /path/to/CANN --destination /new/private/build-dir`

This builds the convolution component by default, using the installed SDK and
current Python's PyTorch and torch_npu. The `--operator` option can also select
`gamma6-rgdr`, `gamma6-rgdr-reference`, `gamma6-rgdr-packed-reference`, or
`gamma8-rgdr`, as well as `conv-reference`, `conv-stride`, and `conv-edge`.
Building does not validate or enable a component. To validate the
convolution's
fresh build against the selected installed backend, run:

`vllm-hust-vspec-validate-tail-conv --component-dir /new/private/build-dir --backend-package /path/to/installed/backend/package`

This test requires idle devices and writes only local component evidence. Its CANN-derived
headers retain their original notices and the adjacent `CANN-LICENSE` applies
to those files; it restricts usage and redistribution to the specified Huawei
AI processor/Software purposes. Other components still need their source and
appropriate licenses. Current experimental paths still depend on additional
host-side backend integration, build tooling and locally generated validation
artifacts. Packaging
the remaining dependencies and providing safe automatic build, installation,
validation and version matching remain required before a formal release.

The prototype proof checks now bind a proof to the two physical devices
reported by `npu-smi` in logical order instead of requiring historical device
numbers. They reject a different allocation and ambiguous device remapping;
they do not themselves replace independent component tests. The experimental
mask component can create its own local proof with:

`vllm-hust-vspec-validate-mask --backend /path/to/prepared/backend --destination /new/private/proof-dir`

This runs a two-device test and writes a private proof only after the numerical,
source-integrity and per-case speed gates pass. It is not a model, task-quality
or end-to-end performance validation. The two GDN-input components have an
experimental local test command:

`vllm-hust-vspec-validate-gdn-input --width 6 --backend /path/to/installed/site-packages --vllm-source /path/to/vllm --destination /new/private/proof-dir`

Run it separately with width `8` and a different destination for the second
component. It rejects an active backend or vLLM source that differs from the
selected paths, and validates the packaged kernel against that backend. These
commands do not validate the full model. The gamma-six reference and gamma-eight
recurrent builds have a separate local proof command:

`vllm-hust-vspec-validate-rgdr --component gamma8 --component-dir /new/private/build-dir --backend-package /path/to/installed/backend/package`

The gamma-six reference uses `--component gamma6-reference`. The packed
reference uses `--component gamma6-packed-reference --reference-dir <validated-reference>`;
the tagged variant uses `--component gamma6-packed --reference-dir <validated-reference>
--prior-dir <validated-packed-reference>`. These tests require the preceding
local proofs and check the admitted source transformations. Other components
also need their own proofs. The reference, stride and edge convolution
components use `vllm-hust-vspec-validate-conv`; the stride test requires
`--reference-dir` pointing to the independently validated reference build.
The prefix bridge has a separate local component test,
`vllm-hust-vspec-validate-prefix-clip --backend /path/to/installed/site-packages --destination /new/private/proof-dir`.

The FIA host adapter can be built in a new private directory with
`vllm-hust-vspec-build-operator --operator fia-host --cann-root /path/to/cann --destination /new/private/fia-build`,
then checked on both allocated devices with
`vllm-hust-vspec-validate-fia-host --directory /new/private/fia-build`.
The two-bank embedding component uses
`vllm-hust-vspec-validate-two-bank --model /path/to/checkpoint --destination /new/private/proof-dir`.
These commands generate only local component evidence; they are not model-level
correctness or performance evidence.

No model, dataset, private benchmark configuration, logs, results or proof
artifacts are distributed in this bundle. Hardware and driver, CANN SDK,
compatible Python/PyTorch/torch_npu and build tools must be available on the
host. Since the full build and NPU correctness/performance checks have not
been performed on this packaged variant, do not use it as a production backend.

For an installed backend, run commands through
`vllm-hust-vspec-run-with-backend --backend-package /path/to/site-packages -- <command> ...`
to select its bundled custom operator directory and library before process
startup. The CANN-generated `set_env.bash` can contain the original build path;
do not rely on it after moving or installing the package elsewhere.
