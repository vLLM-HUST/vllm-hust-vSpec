"""Opt-in startup bridge shared by native and dynamic Qwen3.5 MTP runs.

The launcher adds this directory to PYTHONPATH only with
``--qwen35-mtp-compat``. Python loads this file in every spawned worker before
vLLM or the optional vSpec plugin. Neither bridge imports vllm_hust_vspec.
"""

import os

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
