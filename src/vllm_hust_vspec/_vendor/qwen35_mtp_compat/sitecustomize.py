"""Opt-in startup bridge shared by native and dynamic Qwen3.5 MTP runs.

The launcher adds this directory to PYTHONPATH only with
``--qwen35-mtp-compat``. Python loads this file in every spawned worker before
vLLM or the optional vSpec plugin. Neither bridge imports vllm_hust_vspec.
"""

import os
import sys


_CANN_COMPILER_ENTRYPOINTS = {
    ("op_compile_tool", "op_compiler_tool.py"),
    ("opc_tool", "opc.py"),
}
_script = os.path.normpath(sys.argv[0])
_entrypoint = (os.path.basename(os.path.dirname(_script)), os.path.basename(_script))
_static_installer = (
    _entrypoint == ("scripts", "install.py")
    and ("static_kernel", "ai_core") in zip(
        _script.split(os.sep), _script.split(os.sep)[1:]
    )
)
if _entrypoint in _CANN_COMPILER_ENTRYPOINTS or _static_installer:
    # The compiler starts further Python tools with its environment. Neither
    # it, its children, nor the generated static-kernel package installer are
    # vLLM workers. Keep the parent worker's bridge and fail-closed behavior.
    os.environ.pop("QWEN35_MTP_COMPAT", None)

if os.environ.get("QWEN35_MTP_COMPAT") == "1":
    try:
        import qwen35_mtp_kv_compat
        import qwen35_mtp_mamba_compat

        qwen35_mtp_kv_compat.install()
        qwen35_mtp_mamba_compat.apply_mamba_runtime_compatibility_patch()
    except Exception as exc:
        # Python's site module otherwise logs and continues after errors in
        # sitecustomize, silently starting a server without the matched bridge.
        raise SystemExit(f"Qwen3.5 MTP compatibility startup failed: {exc}") from exc
