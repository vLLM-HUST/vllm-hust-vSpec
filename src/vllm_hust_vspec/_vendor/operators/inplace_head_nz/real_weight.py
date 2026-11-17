"""Read public checkpoint only; use actual balanced proposal ID functions."""
import ast
import hashlib
import importlib.resources
import json
import os
from pathlib import Path
import struct

MODEL = Path(os.environ['VSPEC_MODEL_DIR'])
HEAD_SOURCE = Path(str(importlib.resources.files('vllm_hust_vspec') / 'backends/mtp_active_head.py'))


def load_weights(rank):
    import torch
    from safetensors import safe_open
    index = MODEL / 'model.safetensors.index.json'
    shard = MODEL / json.loads(index.read_text())['weight_map']['lm_head.weight']
    with shard.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]; header = stream.read(length)
    info = json.loads(header)['lm_head.weight']
    assert info['dtype'] == 'BF16' and info['shape'] == [248320, 2048]
    tree = ast.parse(HEAD_SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('active_ids_for_shard', 'balanced_active_ids')]
    assert len(nodes) == 2
    namespace = {}; exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HEAD_SOURCE), 'exec'), namespace)
    ids = namespace['balanced_active_ids'](248320, 98304, rank)
    assert len(ids) == len(set(ids)) == 49152 and ids == sorted(ids)
    with safe_open(str(shard), framework='pt', device='cpu') as stream:
        full = stream.get_tensor('lm_head.weight')
        assert full.shape == (248320, 2048) and full.dtype == torch.bfloat16
        target = full[rank * 124160:(rank + 1) * 124160].contiguous()
        draft = full.index_select(0, torch.tensor(ids, dtype=torch.int64)).contiguous()
    def digest(value):
        return hashlib.sha256(memoryview(value.view(torch.int16).numpy()).cast('B')).hexdigest()
    return dict(draft=draft, target=target), dict(shard=str(shard), key='lm_head.weight', rank=rank,
        tensor_shape=info['shape'], dtype='BF16', header_sha256=hashlib.sha256(header).hexdigest(),
        index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        actual_balanced_ID_source_sha256=hashlib.sha256(HEAD_SOURCE.read_bytes()).hexdigest(),
        target_shard_sha256=digest(target), balanced_draft_sha256=digest(draft),
        global_draft_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        only_public_weight_read_no_shared_writes=True)
