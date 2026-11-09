"""Normal bundled package initialization plus named read-only methods only."""
import torch_npu
from npugraph_ex import CompilerConfig as _BundledCompilerConfig
from .vspec_readonly_compiler import ReadOnlyCompilerWorkerObserver
class ReadOnlyBundledNGEWorkerObserver(ReadOnlyCompilerWorkerObserver):
    pass
