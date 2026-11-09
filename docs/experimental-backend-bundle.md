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
selected or registered by this staging operation. Current experimental paths
still depend on additional host-side backend integration, build tooling and
locally generated validation artifacts. Some CANN-derived headers and build
modules use their own license and are not included in this bundle. Packaging
the remaining dependencies and providing safe automatic build, installation,
validation and version matching remain required before a formal release.

No model, dataset, private benchmark configuration, logs, results or proof
artifacts are distributed in this bundle. Hardware and driver, CANN SDK,
compatible Python/PyTorch/torch_npu and build tools must be available on the
host. Since the full build and NPU correctness/performance checks have not
been performed on this packaged variant, do not use it as a production backend.
