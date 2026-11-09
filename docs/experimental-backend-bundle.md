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
enabled and retain fail-closed validation requirements. A single experimental
Ascend convolution's sources and build entry point are included:

`vllm-hust-vspec-build-operator --cann-root /path/to/CANN --destination /new/private/build-dir`

This builds only that component, using the installed SDK and current Python's
PyTorch and torch_npu. It neither validates nor enables it. To validate its
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
or end-to-end performance validation. Other components need their own proofs.

No model, dataset, private benchmark configuration, logs, results or proof
artifacts are distributed in this bundle. Hardware and driver, CANN SDK,
compatible Python/PyTorch/torch_npu and build tools must be available on the
host. Since the full build and NPU correctness/performance checks have not
been performed on this packaged variant, do not use it as a production backend.
