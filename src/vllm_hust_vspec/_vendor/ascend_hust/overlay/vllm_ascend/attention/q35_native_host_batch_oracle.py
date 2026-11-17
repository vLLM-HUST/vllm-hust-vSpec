"""Independently trace ORIGINAL native parameter derivation, no device calls."""
import ast
import os
from pathlib import Path

SOURCE = Path(os.environ.get('VSPEC_NATIVE_FIA_ORACLE_SOURCE', '/__vspec_oracle_not_configured__'))


class TraceOnly(ast.NodeTransformer):
    def __init__(self):
        self.removed, self.replaced = [], 0

    def visit_Attribute(self, node):
        if ast.unparse(node) == '_EXTRA_CTX.is_draft_model':
            return ast.copy_location(ast.Name(id='is_draft', ctx=ast.Load()), node)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in ('torch.npu.graph_task_update_begin', 'torch.npu.graph_task_update_end', 'event.record'):
                self.removed.append(name)
                return None
            if name == 'torch_npu.npu_fused_infer_attention_score.out':
                self.replaced += 1
                values = ast.Call(func=ast.Name(id='dict', ctx=ast.Load()), args=[], keywords=node.value.keywords)
                return ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='trace', ctx=ast.Load()),
                    attr='append', ctx=ast.Load()), args=[values], keywords=[]))
        return self.generic_visit(node)


def make_resolver():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendAttentionBackendImpl')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'update_graph_params')
    loops = [n for n in ast.walk(method) if isinstance(n, ast.For) and any(
        isinstance(a, ast.Assign) and isinstance(a.value, ast.Name) and a.value.id == 'param'
        and any(isinstance(t, ast.Tuple) and len(t.elts) == 22 for t in a.targets) for a in n.body)]
    assert len(loops) == 1
    transform = TraceOnly()
    loop = transform.visit(loops[0])
    assert transform.replaced == 1
    assert sorted(transform.removed) == ['event.record', 'torch.npu.graph_task_update_begin', 'torch.npu.graph_task_update_end']
    prefix = ast.parse('def resolve(attn_keys, captured_attn_params, handles, events, attn_metadata, draft_attn_key_steps, is_draft, workspace):\n    trace = []\n    attn_count = 0\n    layer_count = 0\n    return trace\n').body[0]
    prefix.body.insert(-1, loop)
    namespace = dict(PagedAttentionGraphParam=type('UnsupportedPagedAttention', (), {}))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])),
                 '<original-native-FIA-argument-trace>', 'exec'), namespace)
    return namespace['resolve']
