"""Private static compiler setup and named read-only post-timer state RPC."""
import os
from pathlib import Path
from .vspec_readonly_nge_worker import ReadOnlyBundledNGEWorkerObserver
from npugraph_ex._acl_concrete_graph import static_kernel as _static
_ROOT=Path(os.environ['HUST_VSPEC_PRIVATE_COMPILER_ROOT']).resolve()
assert _ROOT != _ROOT.parent and _ROOT.is_dir() and Path.cwd().resolve()==_ROOT
assert _static._use_process_isolation,'Refuse non-private global static kernel installation'
_static._compute_opc_compile_jobs=lambda:'8'
class ReadOnlyStaticKernelWorkerObserver(ReadOnlyBundledNGEWorkerObserver):
    def get_vspec_readonly_static_kernel_state(self):
        path=_static._current_install_path
        assert path is not None and Path(path).resolve().is_relative_to(_ROOT)
        assert _static._use_process_isolation and _static._installed_run_pkgs
        return dict(pid=os.getpid(),process_isolation=True,private_install_path=str(Path(path).resolve()),
            installed_package_hashes=sorted(_static._installed_run_pkgs),offline_compiler_jobs=8,
            Python_metadata_only_no_tensor_reads_writes_or_resets=True)
