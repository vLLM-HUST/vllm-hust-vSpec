"""Configure process-isolated native static kernels in a private run directory."""

import os
from pathlib import Path

from npugraph_ex._acl_concrete_graph import static_kernel as _static
from vspec_readonly_nge_worker import ReadOnlyBundledNGEWorkerObserver

_root_value = os.environ["HUST_VSPEC_PRIVATE_COMPILER_ROOT"]
if not Path(_root_value).is_absolute():
    raise RuntimeError("Static compiler root must be absolute")
_ROOT = Path(_root_value).resolve(strict=True)
if (
    not _ROOT.is_dir()
    or _ROOT == _ROOT.parent
    or Path.cwd().resolve() != _ROOT
    or Path(__file__).resolve().is_relative_to(_ROOT)
):
    raise RuntimeError("Static compiler root must be the private working directory")
if not _static._use_process_isolation:
    raise RuntimeError("Native static kernels require process isolation")

# The same bounded offline compile setting used in the dynamic arm.
_static._compute_opc_compile_jobs = lambda: "8"


class ReadOnlyStaticKernelWorkerObserver(ReadOnlyBundledNGEWorkerObserver):
    def get_vspec_readonly_static_kernel_state(self):
        path = _static._current_install_path
        if path is None or not Path(path).resolve().is_relative_to(_ROOT):
            raise RuntimeError("Native static install path is outside the private run")
        if not _static._use_process_isolation:
            raise RuntimeError("Native static process isolation was disabled")
        return {
            "pid": os.getpid(),
            "process_isolation": True,
            "private_install_path": str(Path(path).resolve()),
            "installed_package_hashes": sorted(_static._installed_run_pkgs),
            "offline_compiler_jobs": int(_static._compute_opc_compile_jobs()),
            "Python_metadata_only_no_tensor_reads_writes_or_resets": True,
        }
