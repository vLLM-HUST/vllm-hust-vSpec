"""Exact repository split function plus actual native NPU norm/gating kernels."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(os.environ['VSPEC_VLLM_SOURCE_ROOT']) / 'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'
node = next(node for node in ast.walk(ast.parse(SOURCE.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == 'rearrange_mixed_qkv')
namespace = dict(torch=torch)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
SPLIT = namespace['rearrange_mixed_qkv']


def native_split(mixed):
    return SPLIT(SimpleNamespace(key_dim=2048, value_dim=4096, tp_size=2, head_k_dim=128, head_v_dim=128), mixed)


def native_inputs(mixed, a, b, a_log, dt_bias):
    # Resolve the globals that the actual patched GDN core calls. Its norm
    # comes from vLLM's bundled FLA module, not Ascend's auxiliary norm.
    from vllm_ascend.ops import gdn
    q, k, v = native_split(mixed)
    q, k = gdn.l2norm_fwd(q), gdn.l2norm_fwd(k)
    g, beta = gdn.DeviceOperator.fused_gdn_gating(a_log, a, b, dt_bias)
    return q, k, v, g, beta
