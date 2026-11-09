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
from vllm_hust_vspec.backends.experimental_paths import artifact_dir

PRIMITIVE = artifact_dir('VSPEC_NATIVE_STRIDE_CONV_DIR')
PROOF = PRIMITIVE / 'native-stride-original-and-copy-controls-r1-primitive-audit.json'
LIBRARY = PRIMITIVE / 'build-r1/lib/libvspec_conv_stride_adapter.so'
EDGE = artifact_dir('VSPEC_EDGE_CONV_DIR')
EDGE_PROOF = EDGE / 'runs/even-edge-c46-r1.primitive-audit.json'
EDGE_LIBRARY = EDGE / 'build-r1/lib/libvspec_even_edge_adapter.so'
_WARMED = set()


def verify_primitive():
    data = json.loads(PROOF.read_text())
    assert data['passed'] and data['exact'] and data['cases_per_rank'] == 480
    assert {r['rank'] for r in data['ranks']} == {0, 1}
    for proof in (data, *data['ranks']):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in proof['sha256'].items())
    assert all(r['actual_replay_witness_folded_inside_same_kernel_all_cells_checked']
               for rank in data['ranks'] for r in rank['records'])
    assert all(r['candidate_uses_original_physical_input_stride_without_contiguous_copy']
               and r['entire_qkvz_source_base_including_unused_z_NaN_and_signed_zero_immutable']
               for rank in data['ranks'] for r in rank['records'])


def verify_edge_primitive():
    data = json.loads(EDGE_PROOF.read_text())
    assert data['passed'] and data['exact'] and data['physical_npus'] == [4, 6]
    assert {r['rank'] for r in data['ranks']} == {0, 1}
    for proof in (data, *data['ranks']):
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in proof['sha256'].items())
    for rank in data['ranks']:
        assert len(rank['records']) == 576
        for row in rank['records']:
            assert row['eight_eager_and_graph_revisions_bits_exact']
            assert row['FULL_native_physical_state_bits_exact']
            assert row['independent_native_graph_passed']
            assert row['whole_source_base_weights_metadata_readonly']
            assert row['all_folded_replay_counter_cells_exact']


def transformed(original_tree):
    result = copy.deepcopy(original_tree)
    # The max8 model owns the original full11-slot convolution bank, not9.
    # Broaden ONLY this outer shape eligibility; retain all original shadow code.
    tests = [n for n in ast.walk(result) if isinstance(n, ast.Compare)
             and ast.unparse(n) == 'tuple(state.shape[-2:]) != (9, 4096)']
    assert len(tests) == 1
    tests[0].ops = [ast.NotIn()]
    tests[0].comparators = [ast.parse('((9, 4096), (11, 4096))', mode='eval').body]
    widths = [n for n in ast.walk(result) if isinstance(n, ast.Compare)
        and ast.unparse(n) == 'value.shape[0] not in (meta.num_spec_decodes * 5, meta.num_spec_decodes * 7)']
    assert len(widths) == 1
    widths[0].comparators = [ast.parse('(meta.num_spec_decodes * 3, meta.num_spec_decodes * 5, meta.num_spec_decodes * 7, meta.num_spec_decodes * 9)', mode='eval').body]
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
    if not (weights is owner.weights and bias is None and activation == 1
            and not meta.num_prefills and not meta.num_decodes and meta.num_spec_decodes == 14
            and value.ndim == 2 and value.shape in ((70, 4096), (98, 4096))
            and output.shape == value.shape and weights.shape == (4, 4096)
            and state.ndim == 3 and state.shape[0] >= 100 and state.shape[1:] in ((9, 4096), (11, 4096))
            and query is not None and indices is not None and accepted is not None
            and query.shape == (15,) and indices.shape in ((14, 5), (14, 7), (14, 9))
            and accepted.shape == (14,) and indices.shape[1] >= value.shape[0] // 14):
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
    edge = getattr(owner, '_q35_edge_conv_helper', None)
    if edge is not None and edge_eligible(owner, output, value, weights, state, bias, query, indices, accepted, meta, activation):
        edge.apply(output, value, weights, state, query, indices, accepted)
        return True
    helper = getattr(owner, '_q35_native_conv_tile_helper', None)
    if helper is None or weights is not owner.weights or not eligible(owner, output, value, weights, state, bias,
                                      query, indices, accepted, meta, activation):
        return False
    helper.apply(output, value, weights, state, query, indices, accepted)
    return True


def edge_eligible(owner, output, value, weights, state, bias, query, indices, accepted, meta, activation):
    n = meta.num_spec_decodes
    if not (weights is owner.weights and bias is None and activation == 1
        and not meta.num_prefills and not meta.num_decodes and 1 <= n <= 16
        and value.ndim == 2 and value.shape in ((n * 3, 4096), (n * 9, 4096))
        and output.shape == value.shape and weights.shape == (4, 4096)
        and state.ndim == 3 and state.shape[0] >= n * 9 + 2 and state.shape[1:] == (11, 4096)
        and query is not None and indices is not None and accepted is not None
        and query.shape == (n + 1,) and indices.shape == (n, 9) and accepted.shape == (n,)):
        return False
    if (query.dtype, indices.dtype, accepted.dtype) not in (
        (torch.int32, torch.int32, torch.int32), (torch.int64, torch.int64, torch.int64),
        (torch.int64, torch.int32, torch.int64)):
        return False
    return bool(value.device.type == 'npu'
        and all(t.dtype == torch.bfloat16 for t in (output, value, weights, state))
        and value.stride() in ((4096, 1), (6144, 1))
        and all(t.is_contiguous() and t.device == value.device
            for t in (output, weights, state, query, indices, accepted)))


class EdgeConv:
    def __init__(self, owner):
        assert owner.mode in ('on', 'validate')
        self.owner = owner
        self.counts = torch.zeros(34, dtype=torch.int32, device=owner.source.device)
        self.captured = set()

    def apply(self, output, value, weights, state, queries, indices, accepted):
        # The SAME tested native arithmetic, with its own folded witness.
        torch.ops.vspec_even_edge_conv.project(output, value, weights, state, queries, indices,
            accepted, self.counts, 2048)
        self.captured.add((accepted.numel(), value.shape[0] // accepted.numel()))

    def reset(self):
        self.counts.zero_()

    def metrics(self):
        counts = self.counts.cpu().tolist()
        return dict(captured_shapes=sorted(self.captured),
            actual_replay_counts_by_gamma_and_requests={f'{2 if i < 17 else 8}x{i % 17}': n
                for i, n in enumerate(counts) if n},
            real_weight_native_output_and_full_state_checks=self.owner._edge_weight_receipt,
            original_gamma4_and6_kernel_dispatch_unchanged=True,
            stats_read_after_timers_only=True)


@torch.inference_mode()
def real_edge_weight_admission(owner):
    weights = owner.weights
    original = owner.source.view(4096, 4).transpose(0, 1)
    assert torch.equal(weights.view(torch.int16), original.view(torch.int16))
    records = []
    for n in (1, 14, 16):
        for width in (3, 9):
            tokens = n * width
            source = torch.arange(tokens * 6144, dtype=torch.float32, device=weights.device).reshape(tokens, 6144)
            source = ((source.remainder(127) - 63) / 128).to(torch.bfloat16)
            source[:, 4096:].view(torch.int16).fill_(32705)
            value = source[:, :4096]
            initial = torch.full((192, 11, 4096), 0.03125, dtype=torch.bfloat16, device=weights.device)
            initial.view(torch.int16)[191].fill_(32705)
            initial.view(torch.int16)[190, -1, -1] = -32768
            state, native_state = initial.clone(), initial.clone()
            output, expected = torch.empty_like(value), torch.empty_like(value)
            query = torch.arange(n + 1, dtype=torch.int32, device=weights.device) * width
            indices = torch.arange(n * 9, dtype=torch.int32, device=weights.device).reshape(n, 9)
            accepted = torch.arange(n, dtype=torch.int32, device=weights.device).remainder(width) + 1
            inputs = (source, weights, original, query, indices, accepted)
            snapshots = [t.clone() for t in inputs]
            counts = torch.zeros(34, dtype=torch.int32, device=weights.device)
            torch.ops._C_ascend.npu_causal_conv1d_custom(expected, value, original,
                conv_state=native_state, bias_opt=None, query_start_loc_opt=query,
                cache_indices_opt=indices, initial_state_mode_opt=None,
                num_accepted_tokens_opt=accepted, activation_mode=1, pad_slot_id=-1, run_mode=1)
            torch.ops.vspec_even_edge_conv.project(output, value, weights, state, query, indices, accepted, counts, 2048)
            torch.npu.synchronize()
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
            assert torch.equal(state.view(torch.int16), native_state.view(torch.int16))
            assert all(torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a,
                b.view(torch.int16) if b.dtype == torch.bfloat16 else b) for a, b in zip(inputs, snapshots))
            wanted = torch.zeros_like(counts);wanted[n + (17 if width == 9 else 0)] = 1
            assert torch.equal(counts, wanted)
            records.append(dict(requests=n, width=width, tile=2048,
                actual_weight_output_and_entire_physical_state_bits_exact=True,
                complete_source_weight_and_metadata_readonly=True))
    return records


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
        torch.ops.vspec_conv_strided.project(output, value, weights, state, queries, indices,
                                         accepted, self.counts, 2048)
        shape = value.shape[0]
        if shape not in self.captured:
            self.captured.add(shape)
            print('[q35-native-conv-tile] ' + json.dumps(dict(pid=os.getpid(),
                layer=self.owner.attention.prefix, mode=self.owner.mode, tokens=shape,
                tile=2048, physical_input_stride=list(value.stride()), no_contiguous_input_copy=True,
                original_native_shadow_retained=True,
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
            load_time_real_weight_native_and_full_physical_state_checks=getattr(self.owner, '_native_stride_weight_receipt', None),
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
    state = torch.zeros((192, 11, 4096), dtype=torch.bfloat16, device=weights.device)
    counts = torch.zeros(34, dtype=torch.int32, device=weights.device)
    for width in (5, 7):
        value = torch.zeros((14 * width, 6144), dtype=torch.bfloat16, device=weights.device)[:, :4096]
        output = torch.empty_like(value)
        queries = torch.arange(15, dtype=torch.int32, device=weights.device) * width
        indices = torch.arange(126, dtype=torch.int32, device=weights.device).reshape(14, 9)
        accepted = torch.ones(14, dtype=torch.int32, device=weights.device)
        torch.ops.vspec_conv_strided.project(output, value, weights, state, queries, indices, accepted, counts, 2048)
    torch.npu.synchronize()
    _WARMED.add(key)


@torch.inference_mode()
def real_weight_admission(owner):
    """Outside serving/graph timers: all actual target weights against native."""
    weights = owner.weights
    original_weights = owner.source.view(4096, 4).transpose(0, 1)
    assert torch.equal(weights.view(torch.int16), original_weights.view(torch.int16))
    records = []
    for width in (5, 7):
        tokens = 14 * width
        source = torch.arange(tokens * 6144, dtype=torch.float32, device=weights.device).reshape(tokens, 6144)
        source = ((source.remainder(127) - 63) / 128).to(torch.bfloat16)
        source[:, 4096:].view(torch.int16).fill_(32705)
        value = source[:, :4096]
        source_before = source.clone()
        weights_before, original_before = weights.clone(), original_weights.clone()
        initial = torch.full((192, 11, 4096), 0.03125, dtype=torch.bfloat16, device=weights.device)
        initial.view(torch.int16)[191].fill_(32705)
        initial.view(torch.int16)[190, -1, -1] = -32768
        state, native_state = initial.clone(), initial.clone()
        output, expected = torch.empty_like(value), torch.empty_like(value)
        queries = torch.arange(15, dtype=torch.int32, device=weights.device) * width
        indices = torch.arange(126, dtype=torch.int32, device=weights.device).reshape(14, 9)
        accepted = torch.arange(14, dtype=torch.int32, device=weights.device).remainder(width) + 1
        before_metadata = [x.clone() for x in (queries, indices, accepted)]
        counts = torch.zeros(34, dtype=torch.int32, device=weights.device)
        torch.ops._C_ascend.npu_causal_conv1d_custom(expected, value, original_weights,
            conv_state=native_state, bias_opt=None, query_start_loc_opt=queries,
            cache_indices_opt=indices, initial_state_mode_opt=None,
            num_accepted_tokens_opt=accepted, activation_mode=1, pad_slot_id=-1, run_mode=1)
        torch.ops.vspec_conv_strided.project(output, value, weights, state, queries, indices, accepted, counts, 2048)
        torch.npu.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(state.view(torch.int16), native_state.view(torch.int16))
        assert torch.equal(source.view(torch.int16), source_before.view(torch.int16))
        assert torch.equal(weights.view(torch.int16), weights_before.view(torch.int16))
        assert torch.equal(original_weights.view(torch.int16), original_before.view(torch.int16))
        assert all(torch.equal(a, b) for a, b in zip((queries, indices, accepted), before_metadata))
        wanted = torch.zeros_like(counts);wanted[14 + (17 if width == 7 else 0)] = 1
        assert torch.equal(counts, wanted)
        records.append(dict(tokens=tokens, input_stride=6144, state_length=11, cache_lines=192, index_stride=9,
            real_original_noncontiguous_weight_and_cached_weight_bits_exact=True,
            output_and_whole_native_physical_state_bits_exact=True,
            complete_inputs_weights_and_metadata_bits_readonly=True))
    return records


def install_native_conv_tile():
    from vllm_ascend.ops.q35_cached_conv import CachedConvWeights
    from vllm_ascend.worker.worker import NPUWorker
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original = CachedConvWeights.apply
    if getattr(original, '_q35_native_conv_tile', False):
        return
    source = Path(inspect.getsourcefile(original))
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError('Unknown cached native convolution apply/source shadow')
    verify_primitive()
    verify_edge_primitive()
    torch.ops.load_library(str(LIBRARY))
    torch.ops.load_library(str(EDGE_LIBRARY))
    torch.library.register_fake('vspec_conv_strided::project')(fake_project)
    torch.library.register_fake('vspec_even_edge_conv::project')(fake_project)
    method = ast.parse(textwrap.dedent(inspect.getsource(original))).body[0]
    replacement = bind(original, transformed(method))
    replacement._q35_native_conv_tile = True
    CachedConvWeights.apply = replacement
    original_init = CachedConvWeights.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        warmup(self.weights)
        self._native_stride_weight_receipt = real_weight_admission(self)
        print('[native-conv-stride-real-weight-admitted] ' + json.dumps(dict(
            pid=os.getpid(),layer=self.attention.prefix,actual_weight_bits_exact=True,
            original_native_output_and_whole192_line_11_slot_state_bits_exact=True,
            widths=[5,7],index_stride=9,input_stride=6144)),flush=True)
        self._q35_native_conv_tile_helper = ConvTile(self)
        self._edge_weight_receipt = real_edge_weight_admission(self)
        self._q35_edge_conv_helper = EdgeConv(self)
        print('[edge-conv-all-shapes-real-weight-admitted] ' + json.dumps(dict(
            pid=os.getpid(), layer=self.attention.prefix, cases=len(self._edge_weight_receipt),
            full_physical_state_and_output_exact=True, unchanged_gamma4_and6=True)), flush=True)

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
            row['native_edge_conv'] = by_name[row['layer']]._q35_edge_conv_helper.metrics()
        return result

    @wraps(original_reset)
    def reset(self):
        result = original_reset(self)
        for owner in helpers(self).values():
            owner._q35_native_conv_tile_helper.reset()
            owner._q35_edge_conv_helper.reset()
        return result

    NPUWorker.get_private_q35_gdn_input_metrics = metrics
    NPUWorker.reset_private_q35_gdn_input_stats = reset
    original_load = NPUModelRunner.load_model

    @wraps(original_load)
    def load_and_require_all_layers(runner, *args, **kwargs):
        result = original_load(runner, *args, **kwargs)
        manager = runner._q35_cached_conv_manager
        assert os.environ['VLLM_ASCEND_Q35_CACHED_CONV'] == 'on'
        assert manager.mode == 'on' and len(manager.layers) == 30
        for owner in manager.layers.values():
            assert hasattr(owner, '_q35_native_conv_tile_helper')
            assert len(owner._native_stride_weight_receipt) == 2
            assert len(owner._edge_weight_receipt) == 6
        print('[native-conv-stride-all30-bound-before-capture] ' + json.dumps(dict(
            pid=os.getpid(),cached_conv_mode=manager.mode,layers=len(manager.layers),
            all_actual_weight_native_preflights_passed=True)),flush=True)
        return result

    NPUModelRunner.load_model = load_and_require_all_layers
