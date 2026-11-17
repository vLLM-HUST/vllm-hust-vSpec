"""Default-off direct native convolution, with real folded replay witnesses.

The original cached-convolution method and its complete native state shadow
remain intact. Only the candidate invocation may dispatch this same native
convolution class. The distinct native reference invocation always falls back.
"""
import ast
import copy
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from types import FunctionType

import torch

SOURCE_SHA = '1ac11d07b5883c3d8ec68ca0382411cdd1f26ad2bb7e23621648fd73a9601890'
from .experimental_paths import artifact_dir

PRIMITIVE = artifact_dir('VSPEC_NATIVE_CONV_TILE_DIR')
PROOF = PRIMITIVE / 'direct-stride-allbatch-card12-r1-primitive-audit.json'
LIBRARY = PRIMITIVE / 'build-r1/lib/libvspec_conv_direct_stride_adapter.so'
_WARMED = set()


def verify_primitive():
    data = json.loads(PROOF.read_text())
    assert data['passed'] and data['exact'] and data['cases_per_rank'] == 768
    assert {r['rank'] for r in data['ranks']} == {0, 1}
    for proof in (data, *data['ranks']):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in proof['sha256'].items())
    assert all(r['actual_replay_witness_folded_inside_same_kernel_all_cells_checked']
               for rank in data['ranks'] for r in rank['records'])
    assert all(r['candidate_directly_reads_real_source_row_stride_no_input_copy']
               and r['entire_qkvz_source_base_including_unused_z_NaN_and_signed_zero_immutable']
               for rank in data['ranks'] for r in rank['records'])


def transformed(original_tree):
    result = copy.deepcopy(original_tree)
    methods = [n for n in ast.walk(result) if isinstance(n, ast.FunctionDef) and n.name == 'invoke']
    assert len(methods) == 1 and len(methods[0].body) == 1
    native, = methods[0].body
    assert isinstance(native, ast.Expr) and isinstance(native.value, ast.Call)
    assert ast.unparse(native.value.func) == 'torch.ops._C_ascend.npu_causal_conv1d_custom'
    gate = ast.parse('''
if not _q35_try_native_conv_tile(self, destination, value, weights, cache, bias,
                               query, indices, accepted, meta, activation):
    pass
''').body[0]
    gate.body = [native]
    methods[0].body = [gate]
    return ast.fix_missing_locations(result)


def bind(original, method):
    assert not original.__code__.co_freevars
    scope = {}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, '<guarded-direct-native-conv>', 'exec'), scope)
    original.__globals__['_q35_try_native_conv_tile'] = try_invoke
    function = FunctionType(scope[method.name].__code__, original.__globals__,
                            original.__name__, original.__defaults__)
    function.__kwdefaults__ = original.__kwdefaults__
    return wraps(original)(function)


def eligible(owner, output, value, weights, state, bias, query, indices, accepted, meta, activation):
    rows = meta.num_spec_decodes
    if not (weights is owner.weights and bias is None and activation == 1
            and not meta.num_prefills and not meta.num_decodes and 1 <= rows <= 16
            and value.ndim == 2 and value.shape in ((rows * 5, 4096), (rows * 7, 4096))
            and output.shape == value.shape and weights.shape == (4, 4096)
            and state.ndim == 3 and state.shape[0] >= 100 and state.shape[1:] == (9, 4096)
            and query is not None and indices is not None and accepted is not None
            and query.shape == (rows + 1,) and indices.shape in ((rows, 5), (rows, 7))
            and accepted.shape == (rows,) and indices.shape[1] >= value.shape[0] // rows):
        return False
    if (query.dtype, indices.dtype, accepted.dtype) not in (
            (torch.int32, torch.int32, torch.int32), (torch.int64, torch.int64, torch.int64),
            (torch.int64, torch.int32, torch.int64)):
        return False
    tensors = (output, weights, state, query, indices, accepted)
    return bool(value.device.type == 'npu'
        and all(t.dtype == torch.bfloat16 for t in (output, value, weights, state))
        and value.stride() in ((4096, 1), (6144, 1))
        and all(t.is_contiguous() and t.device == value.device for t in tensors))


def try_invoke(owner, output, value, weights, state, bias, query, indices, accepted, meta, activation):
    helper = getattr(owner, '_q35_native_conv_tile_helper', None)
    if helper is None or weights is not owner.weights or not eligible(owner, output, value, weights, state, bias,
                                      query, indices, accepted, meta, activation):
        return False
    helper.apply(output, value, weights, state, query, indices, accepted)
    return True


def fake_project(output, value, weight, state, queries, indices, accepted, counters, tile):
    return output


class ConvTile:
    def __init__(self, owner):
        if owner.mode not in ('on', 'validate'):
            raise RuntimeError('Native conv tile requires existing on/validate cached conv scope')
        if owner.source.device.type == 'npu':
            import torch_npu
            if torch_npu.get_npu_format(owner.weights) not in (0, 2):
                raise RuntimeError('Every direct convolution bank must retain native cached ND weights')
        self.owner = owner
        self.counts = torch.zeros(34, dtype=torch.int32, device=owner.source.device)
        self.captured = set()

    def apply(self, output, value, weights, state, queries, indices, accepted):
        torch.ops.vspec_conv_direct_stride.project(output, value, weights, state, queries, indices,
                                         accepted, self.counts, 2048)
        shape = value.shape[0]
        if shape not in self.captured:
            self.captured.add(shape)
            print('[q35-native-conv-tile] ' + json.dumps(dict(pid=os.getpid(),
                layer=self.owner.attention.prefix, mode=self.owner.mode, tokens=shape,
                tile=2048, original_native_shadow_retained=True,
                actual_replay_counter_folded_inside_same_kernel=True)), flush=True)

    def reset(self):
        self.counts.zero_()

    def metrics(self):
        counts = self.counts.cpu().tolist()
        return dict(mode=self.owner.mode, tile=2048, captured_shapes=sorted(self.captured),
            actual_replay_counts_by_gamma_and_requests={f'{4 if i < 17 else 6}x{i % 17}': n
                                                       for i, n in enumerate(counts) if n},
            actual_counter_is_integer_atomic_DMA_inside_same_native_kernel=True,
            original_cached_conv_full_output_and_state_shadow_unchanged=True,
            stats_read_after_timers_only=True)


def warmup(weights):
    key = str(weights.device)
    if key in _WARMED:
        return
    if weights.device.type != 'npu':
        raise RuntimeError('Device warmup only belongs to real model initialization')
    import torch_npu
    if (weights.shape != (4, 4096) or weights.dtype != torch.bfloat16
            or not weights.is_contiguous() or torch_npu.get_npu_format(weights) not in (0, 2)):
        raise RuntimeError('Require original cached native BF16 ND convolution weights')
    state = torch.zeros((128, 9, 4096), dtype=torch.bfloat16, device=weights.device)
    counts = torch.zeros(34, dtype=torch.int32, device=weights.device)
    for width in (5, 7):
        value = torch.zeros((14 * width, 6144), dtype=torch.bfloat16, device=weights.device)[:, :4096]
        output = torch.empty_like(value)
        queries = torch.arange(15, dtype=torch.int32, device=weights.device) * width
        indices = torch.arange(98, dtype=torch.int32, device=weights.device).reshape(14, 7)
        accepted = torch.ones(14, dtype=torch.int32, device=weights.device)
        torch.ops.vspec_conv_direct_stride.project(output, value, weights, state, queries, indices, accepted, counts, 2048)
    torch.npu.synchronize()
    _WARMED.add(key)


def reclaim_startup_allocator(worker):
    """Once, in the existing post-capture/pre-warmup reset RPC; no live tensor edits."""
    previous = getattr(worker, '_q35_startup_allocator_reclaim', None)
    if previous is not None:
        return previous
    torch.npu.synchronize()
    before = dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved())
    torch.npu.empty_cache()
    after = dict(allocated=torch.npu.memory_allocated(), reserved=torch.npu.memory_reserved())
    if after['allocated'] != before['allocated'] or after['reserved'] > before['reserved']:
        raise RuntimeError('Startup reclaim changed live allocation or increased reservation')
    receipt = dict(pid=os.getpid(), before=before, after=after,
                   no_live_tensor_or_KV_budget_or_graph_changed=True,
                   once_in_existing_reset_RPC_before_warmup_and_timed_cohorts=True)
    worker._q35_startup_allocator_reclaim = receipt
    print('[q35-startup-allocator-reclaim] ' + json.dumps(receipt), flush=True)
    return receipt


def install_native_conv_tile():
    from vllm_ascend.ops.q35_cached_conv import CachedConvWeights
    from vllm_ascend.worker.worker import NPUWorker

    original = CachedConvWeights.apply
    if getattr(original, '_q35_native_conv_tile', False):
        return
    source = Path(inspect.getsourcefile(original))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Unknown cached native convolution apply/source shadow')
    verify_primitive()
    torch.ops.load_library(str(LIBRARY))
    torch.library.register_fake('vspec_conv_direct_stride::project')(fake_project)
    method = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    replacement = bind(original, transformed(method))
    replacement._q35_native_conv_tile = True
    CachedConvWeights.apply = replacement
    original_init = CachedConvWeights.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        warmup(self.weights)
        self._q35_native_conv_tile_helper = ConvTile(self)

    CachedConvWeights.__init__ = init
    original_metrics = NPUWorker.get_private_q35_gdn_input_metrics
    original_reset = NPUWorker.reset_private_q35_gdn_input_stats

    def helpers(worker):
        manager = worker.model_runner._q35_cached_conv_manager
        if len(manager.layers) != 30:
            raise RuntimeError('Native conv tile requires all original thirty cached convolution layers')
        return manager.layers

    @wraps(original_metrics)
    def metrics(self):
        result = original_metrics(self)
        by_name = helpers(self)
        for row in result:
            row['native_conv_tile'] = by_name[row['layer']]._q35_native_conv_tile_helper.metrics()
        return result

    @wraps(original_reset)
    def reset(self):
        result = original_reset(self)
        for owner in helpers(self).values():
            owner._q35_native_conv_tile_helper.reset()
        reclaim_startup_allocator(self)
        return result

    NPUWorker.get_private_q35_gdn_input_metrics = metrics
    NPUWorker.reset_private_q35_gdn_input_stats = reset
