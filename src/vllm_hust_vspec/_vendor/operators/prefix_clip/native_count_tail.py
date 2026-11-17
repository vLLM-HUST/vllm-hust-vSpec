"""Exact native GPU count/last-token tail; host backup setup excluded explicitly."""
import ast
import os
from copy import deepcopy
from pathlib import Path
SOURCE=Path(os.environ['VSPEC_ASCEND_SOURCE_ROOT'])/'vllm_ascend/spec_decode/llm_base_proposer.py'

def extract():
    tree=ast.parse(SOURCE.read_text())
    fn=deepcopy(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
        and n.name=='prepare_next_token_ids_padded'))
    first=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='discard_sampled_tokens_req_indices' for t in n.targets))
    fn.body=fn.body[first:];fn.decorator_list=[]
    return ast.fix_missing_locations(fn)

def load():
    import torch
    from vllm_ascend.device.device_op import DeviceOperator
    scope=dict(torch=torch,DeviceOperator=DeviceOperator)
    # Preserve the exact AST operations. Remove annotations only so the
    # independent recipe does not require constructing a full serving engine.
    fn=extract();fn.returns=None
    for a in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs):a.annotation=None
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),scope)
    return scope[fn.name]
