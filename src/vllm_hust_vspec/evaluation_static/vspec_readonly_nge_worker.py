"""Initialize the bundled NPU graph compiler, then expose metadata methods."""

import torch_npu as _torch_npu  # noqa: F401
from npugraph_ex import CompilerConfig as _BundledCompilerConfig  # noqa: F401
from vspec_readonly_compiler import ReadOnlyCompilerWorkerObserver


class ReadOnlyBundledNGEWorkerObserver(ReadOnlyCompilerWorkerObserver):
    pass
