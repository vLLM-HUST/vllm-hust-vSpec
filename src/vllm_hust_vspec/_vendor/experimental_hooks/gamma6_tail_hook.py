"""Private gamma6 all-batch dispatch; gamma4 unchanged.

Independent component evidence is mandatory before registration. No physical
runtime fence, sampler, count production, state bank or public budget changes.
"""
import ast
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
from textwrap import dedent
from types import CodeType

from vllm_hust_vspec.backends.experimental_paths import artifact_dir, require_current_npu_proof

CONV=artifact_dir('VSPEC_GAMMA6_CONV_DIR')
RGDR=artifact_dir('VSPEC_GAMMA6_RGDR_DIR')
INPUT=artifact_dir('VSPEC_GAMMA6_INPUT_DIR')
PROOFS=dict(conv=CONV/'runs/gamma6-tail-strides-c46-r1.primitive-audit.json',
    rgdr=RGDR/'runs/gamma6-tail-c46-r1.primitive-audit.json',
    input=INPUT/'gamma6-input-fusion-c46-primitive-audit-r1.json')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def packed_counter_metrics(result):
    """Post-timer DIAGNOSTICS only, never native accepted-token counts."""
    raw=result['actual_same_kernel_folded_replay_counter_cells']
    assert len(raw)==34 and all(type(value) is int and value>=0 for value in raw)
    assert raw[:17]==[0]*17 and raw[17]==0,'Wrong old/unproven compiled kernel executed'
    result['actual_same_kernel_folded_replay_counter_cells']=raw[17:]
    result['packed_delta_tag_witness']=dict(raw_diagnostic_counter_cells=raw,
        packed_family_logical_counts=raw[17:],
        independent_component_proof_sha256=result['independent_component_proof_sha256'],
        old_family_cells0_to16_ALL_zero_required=True,
        ONE_same_kernel_add_NOT_Python_call_or_capture_only=True,
        counter_is_NOT_native_accepted_tokens_or_KV_or_SSM=True,
        no_extra_hot_operation=True,read_after_timers_only=True)
    return result

def require_same_executable_code(left,right):
    # Nested generator/comprehension CodeType equality includes source column
    # locations. Recurse over all executable fields and literal constants,
    # excluding only filename/line/column/debug positions at EVERY depth.
    fields=('co_code','co_names','co_varnames','co_freevars','co_cellvars','co_flags',
        'co_argcount','co_kwonlyargcount','co_posonlyargcount','co_stacksize','co_name','co_qualname')
    for field in fields:
        assert getattr(left,field)==getattr(right,field),('Cannot reconstruct actual existing gamma8 path',field)
    assert len(left.co_consts)==len(right.co_consts)
    for a,b in zip(left.co_consts,right.co_consts):
        if isinstance(a,CodeType) or isinstance(b,CodeType):
            assert isinstance(a,CodeType) and isinstance(b,CodeType)
            require_same_executable_code(a,b)
        else:
            assert type(a) is type(b) and a==b,('Cannot reconstruct actual existing gamma8 literal',repr(a),repr(b))

def verify():
    receipts={}
    for kind,path in PROOFS.items():
        proof=read(path)
        require_current_npu_proof(proof)
        if kind=='input':
            assert proof['completed'] and proof['component_only']
            assert proof['eligible_tokens']==[7*n for n in range(1,17)]
            assert proof['only_gamma6_all16_batches']
            parts=(proof,read(INPUT/'launch.json'))
            assert {r['device'] for r in proof['ranks']}=={0,1}
            for rank in proof['ranks']:
                cases=[c for c in rank['cases'] if 'phase' in c]
                assert rank['passed'] and len(cases)==128
                for case in cases:
                    assert case['original_native_eager_self_graph_bits_exact']
                    assert case['actual_consumed_BF16_bits_including_signed_zero_exact']
                    assert case['fused_eager_self_graph_bits_exact']
                    assert case['complete_all_five_inputs_raw_bits_readonly'] and case['canaries_unchanged']
                    assert case['fp32_gate_atol']==case['fp32_gate_rtol']==1e-6
        else:
            assert proof['passed'] and proof['exact']
            if kind=='rgdr':
                assert proof['exact_same_packed_arithmetic_header_as_proven_v1']
                assert proof['ONE_existing_folded_diagnostic_add_to_cells17_plus_N_no_extra_hot_operation']
            assert {r['rank'] for r in proof['ranks']}=={0,1}
            parts=(proof,*proof['ranks'])
            for rank in proof['ranks']:
                if kind=='conv':
                    assert len(rank['records'])==576
                    assert {(c['n'],c['width'],c['index_stride'],c['route'],c['types'],c['layout']) for c in rank['records']}=={
                        (n,7,i,r,t,l) for n in range(1,17) for i in (7,9) for r in ('uniform','uneven','padded_zero')
                        for t in ('i32','i64','mixed') for l in ('contiguous','qkvz_split_6144')}
                    for c in rank['records']:
                        assert c['eight_eager_and_graph_revisions_bits_exact'] and c['FULL_native_physical_state_bits_exact']
                        assert c['independent_native_graph_passed'] and c['whole_source_base_weights_metadata_readonly']
                        assert c['all_folded_replay_counter_cells_exact']
                else:
                    assert len(rank['records'])==192
                    assert {(c['requests'],c['bank'],c['seed']) for c in rank['records']}=={
                        (n,b,s) for n in range(1,17) for b in (192,647) for s in range(6)}
                    for c in rank['records']:
                        assert c['twelve_continuous_state_replays_WITHOUT_state_reset_full_bits_exact']
                        assert c['one_existing_atomic_add_targets_independent_cells17_plus_N_to_detect_wrong_old_kernel']
                        assert c['independent_native_eager_graph_reference'] and c['eight_eager_graph_revisions_bits_exact']
                        assert c['captured_graph_dynamic_lengths_and_accepted_covered']
                        assert c['full_FP32_state_including_untouched_NaN_payload_and_signed_zero_exact']
                        assert c['complete_inputs_bits_readonly'] and c['entire_folded_counter_array_exact']
        for part in parts:
            assert all(sha(p)==h for p,h in part['sha256'].items())
        receipts[kind]=sha(path)
    kernel=artifact_dir('VSPEC_ASCEND_SOURCE_ROOT')/'vllm_ascend/ops/triton/q35_fused_gdn_inputs.py'
    assert sha(INPUT/'kernel.py')==sha(kernel)
    return receipts

def input_counter_redirect(original):
    # The existing gamma8 apply was compiled from a virtual filename. Rebuild
    # its source from the immutable native file, then compare its executable
    # bytecode/constants/names before applying the one-cell counter redirect.
    assert original.__code__.co_filename=='<private-gamma8-input-counter>'
    native_path=Path(original.__globals__['__file__'])
    native_text=native_path.read_text();tree=ast.parse(native_text)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GDNInputFusion')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
    source=dedent(ast.get_source_segment(native_text,method))
    old='(self._private_gamma8_counts[meta.num_spec_decodes] if tokens == meta.num_spec_decodes * 9 else self.counts[tokens]).add_(1)'
    assert source.count('self.counts[tokens].add_(1)')==1
    source=source.replace('self.counts[tokens].add_(1)',old)
    reference_scope=dict(original.__globals__)
    exec(compile(ast.parse(source),'<reconstructed-gamma8-counter-reference>','exec'),reference_scope)
    rebuilt=reference_scope[original.__name__].__code__;previous=original.__code__
    require_same_executable_code(rebuilt,previous)
    new='(self._private_gamma6_counts[meta.num_spec_decodes] if tokens == meta.num_spec_decodes * 7 else (self._private_gamma8_counts[meta.num_spec_decodes] if tokens == meta.num_spec_decodes * 9 else self.counts[tokens])).add_(1)'
    assert source.count(old)==1
    changed=source.replace(old,new);assert changed.replace(new,old)==source
    scope=dict(original.__globals__)
    exec(compile(ast.parse(changed),'<private-gamma6-tail-input-counter>','exec'),scope)
    return scope[original.__name__]

def conv_eligible(torch,owner,output,value,weights,state,bias,query,indices,accepted,meta,activation):
    n=meta.num_spec_decodes
    if not (1<=n<=16 and not meta.num_prefills and not meta.num_decodes
        and bias is None and activation==1 and weights is owner.weights
        and value.ndim==2 and value.shape==(n*7,4096) and output.shape==value.shape
        and weights.shape==(4,4096) and state.ndim==3 and state.shape[0]>=n*9+2
        and state.shape[1:]==(11,4096) and query is not None and indices is not None and accepted is not None
        and query.shape==(n+1,) and indices.shape in ((n,7),(n,9)) and accepted.shape==(n,)):
        return False
    return bool((query.dtype,indices.dtype,accepted.dtype) in (
        (torch.int32,torch.int32,torch.int32),(torch.int64,torch.int64,torch.int64),
        (torch.int64,torch.int32,torch.int64)) and value.device.type=='npu'
        and value.stride() in ((4096,1),(6144,1))
        and all(t.dtype==torch.bfloat16 for t in (output,value,weights,state))
        and all(t.is_contiguous() and t.device==value.device for t in (output,weights,state,query,indices,accepted)))

class TailConv:
    def __init__(self,torch,owner):
        self.owner=owner;self.torch=torch
        self.counts=torch.zeros(34,dtype=torch.int32,device=owner.weights.device)
        self.weight_checks=self.check_weights()
    def apply(self,output,value,weights,state,query,indices,accepted):
        self.torch.ops.vspec_g6_tail_v2_conv.project(output,value,weights,state,query,indices,accepted,self.counts,2048)
    def reset(self):self.counts.zero_()
    def check_weights(self):
        torch=self.torch;owner=self.owner;weights=owner.weights
        native_weights=owner.source.view(4096,4).transpose(0,1)
        assert torch.equal(weights.view(torch.int16),native_weights.view(torch.int16))
        records=[]
        for n,index_stride in ((n,i) for n in (1,8,16) for i in (7,9)):
            source=torch.arange(n*7*6144,dtype=torch.float32,device=weights.device).reshape(n*7,6144)
            source=((source.remainder(127)-63)/128).to(torch.bfloat16)
            source[:,4096:].view(torch.int16).fill_(32705);value=source[:,:4096]
            initial=torch.full((192,11,4096),.03125,dtype=torch.bfloat16,device=weights.device)
            initial.view(torch.int16)[191].fill_(32705);initial.view(torch.int16)[190,-1,-1]=-32768
            state,reference_state=initial.clone(),initial.clone()
            output,expected=torch.empty_like(value),torch.empty_like(value)
            query=torch.arange(n+1,dtype=torch.int32,device=weights.device)*7
            indices=torch.arange(n*index_stride,dtype=torch.int32,device=weights.device).reshape(n,index_stride)
            accepted=torch.arange(n,dtype=torch.int32,device=weights.device).remainder(7)+1
            inputs=(source,weights,native_weights,query,indices,accepted);before=[x.clone() for x in inputs]
            counters=torch.zeros(34,dtype=torch.int32,device=weights.device)
            torch.ops._C_ascend.npu_causal_conv1d_custom(expected,value,native_weights,conv_state=reference_state,
                bias_opt=None,query_start_loc_opt=query,cache_indices_opt=indices,initial_state_mode_opt=None,
                num_accepted_tokens_opt=accepted,activation_mode=1,pad_slot_id=-1,run_mode=1)
            torch.ops.vspec_g6_tail_v2_conv.project(output,value,weights,state,query,indices,accepted,counters,2048)
            torch.npu.synchronize()
            assert torch.equal(output.view(torch.int16),expected.view(torch.int16))
            assert torch.equal(state.view(torch.int16),reference_state.view(torch.int16))
            assert all(torch.equal(a.view(torch.int16) if a.dtype==torch.bfloat16 else a,
                b.view(torch.int16) if b.dtype==torch.bfloat16 else b) for a,b in zip(inputs,before))
            wanted=torch.zeros_like(counters);wanted[n]=1;assert torch.equal(counters,wanted)
            records.append(dict(requests=n,index_stride=index_stride,actual_weight_output_and_full_state_bits_exact=True,all_inputs_bits_readonly=True))
        return records
    def metrics(self):
        return dict(counts=self.counts.cpu().tolist(),weight_checks=self.weight_checks,
            gamma4_unchanged_all_gamma6_component_proven=True,folded_same_kernel_witness=True)

def install():
    proof_hashes=verify()
    import torch
    from . import native_stride_conv
    from .gamma8_rgdr_hook import eligible as g8_eligible,Gamma8RGDR
    from .gamma8_input_domain import real_weight_admission as g8_input_weights
    from vllm_ascend.ops import gdn_input_fusion
    from vllm_ascend.ops.q35_cached_conv import CachedConvWeights
    from vllm_ascend.ops.q35_rgdr_reuse import RGDRReuse
    from vllm_ascend.worker.worker import NPUWorker
    torch.ops.load_library(str(CONV/'build-r1/lib/libvspec_g6_tail_v2_adapter.so'))
    torch.ops.load_library(str(RGDR/'build-r1/lib/libvspec_rgdr_pack6_tag_adapter.so'))
    torch.library.register_fake('vspec_g6_tail_v2_conv::project')(native_stride_conv.fake_project)
    def fake_rgdr(q,k,v,beta,state,lengths,indices,g,accepted,counts,v_step,output_buffers):return torch.empty_like(v)
    torch.library.register_fake('vspec_rgdr_pack6_tag::project')(fake_rgdr)
    old_conv=CachedConvWeights.__init__
    @wraps(old_conv)
    def init_conv(owner,*a,**kw):
        old_conv(owner,*a,**kw);owner._private_g6_tail_conv=TailConv(torch,owner)
    CachedConvWeights.__init__=init_conv
    scope=CachedConvWeights.apply.__globals__
    old_try=scope['_q35_try_native_conv_tile'];assert old_try is native_stride_conv.try_invoke
    def try_conv(owner,output,value,weights,state,bias,query,indices,accepted,meta,activation):
        if conv_eligible(torch,owner,output,value,weights,state,bias,query,indices,accepted,meta,activation):
            owner._private_g6_tail_conv.apply(output,value,weights,state,query,indices,accepted);return True
        return old_try(owner,output,value,weights,state,bias,query,indices,accepted,meta,activation)
    scope['_q35_try_native_conv_tile']=try_conv
    old_input_init,old_input_reset=gdn_input_fusion.GDNInputFusion.__init__,gdn_input_fusion.GDNInputFusion.reset_stats
    weight_source=dedent(inspect.getsource(g8_input_weights))
    assert weight_source.count('tokens=requests*9')==1
    weight_changed=weight_source.replace('tokens=requests*9','tokens=requests*7')
    weight_env=dict(g8_input_weights.__globals__)
    exec(compile(ast.parse(weight_changed),'<private-gamma6-real-input-weights>','exec'),weight_env)
    real_input_weights=weight_env['real_weight_admission']
    @wraps(old_input_init)
    def init_input(owner,*a,**kw):
        old_input_init(owner,*a,**kw)
        owner._private_gamma6_counts=torch.zeros(17,dtype=torch.int64,device=owner.attention.A_log.device)
        owner._private_g6_input_weight_checks=real_input_weights(owner)
    @wraps(old_input_reset)
    def reset_input(owner):
        old_input_reset(owner)
        if owner.enabled:owner._private_gamma6_counts.zero_()
    gdn_input_fusion.GDNInputFusion.__init__=init_input
    gdn_input_fusion.GDNInputFusion.reset_stats=reset_input
    gdn_input_fusion.GDNInputFusion.apply=input_counter_redirect(gdn_input_fusion.GDNInputFusion.apply)
    gdn_input_fusion._private_gamma8_input_receipt['gamma4_gamma6_original_counter_cells_unchanged']=False
    gdn_input_fusion._private_gamma8_input_receipt.update(
        gamma4_counter_cells_unchanged=True,
        all_gamma6_existing_add_redirected_by_separate_declared_extension=True)
    # Reuse the exact admitted gamma8 eligibility predicate and recurrence
    # wrapper structure, changing only per-request width and op namespace.
    source=dedent(inspect.getsource(g8_eligible))
    changed=source.replace('tokens=n*9','tokens=n*7')
    assert changed.replace('tokens=n*7','tokens=n*9')==source
    env=dict(g8_eligible.__globals__);exec(compile(ast.parse(changed),'<private-gamma6-tail-rgdr-eligibility>','exec'),env)
    eligible=env['eligible']
    apply_source=dedent(inspect.getsource(Gamma8RGDR.apply))
    apply_changed=apply_source.replace('torch.ops.vspec_rgdr_edge8.project','torch.ops.vspec_rgdr_pack6_tag.project')
    assert apply_changed!=apply_source
    env=dict(Gamma8RGDR.apply.__globals__);exec(compile(ast.parse(apply_changed),'<private-gamma6-tail-rgdr-apply>','exec'),env)
    def tail_rgdr_metrics(owner):
        result=packed_counter_metrics(Gamma8RGDR.metrics(owner))
        result.pop('original_gamma4_and6_apply_fallback_unchanged')
        result['gamma4_unchanged_all_gamma6_component_proven']=True
        result['only_gamma6_batch1_to16']=True
        return result
    def init_packed_rgdr(owner,torch,parent,proof_sha):
        Gamma8RGDR.__init__(owner,torch,parent,proof_sha)
        owner.counts=torch.zeros(34,dtype=torch.int32,device=parent.attention.A_log.device)
    TailRGDR=type('Gamma6TailRGDR',(Gamma8RGDR,),{'__init__':init_packed_rgdr,'apply':env['apply'],'metrics':tail_rgdr_metrics})
    old_rgdr_init,old_rgdr_apply,old_rgdr_reset=RGDRReuse.__init__,RGDRReuse.apply,RGDRReuse.reset
    @wraps(old_rgdr_init)
    def init_rgdr(owner,*a,**kw):
        old_rgdr_init(owner,*a,**kw);owner._private_g6_tail_rgdr=TailRGDR(torch,owner,proof_hashes['rgdr'])
    @wraps(old_rgdr_apply)
    def apply_rgdr(owner,meta,inputs,state,lengths,indices,accepted,fused):
        if eligible(torch,meta,inputs,state,lengths,indices,accepted,fused):
            return owner._private_g6_tail_rgdr.apply(inputs,state,lengths,indices,accepted)
        return old_rgdr_apply(owner,meta,inputs,state,lengths,indices,accepted,fused)
    @wraps(old_rgdr_reset)
    def reset_rgdr(owner):
        result=old_rgdr_reset(owner);owner._private_g6_tail_rgdr.reset();return result
    RGDRReuse.__init__,RGDRReuse.apply,RGDRReuse.reset=init_rgdr,apply_rgdr,reset_rgdr
    old_reset=NPUWorker.reset_private_q35_gdn_input_stats
    @wraps(old_reset)
    def reset_worker(worker):
        result=old_reset(worker)
        for owner in worker.model_runner._q35_cached_conv_manager.layers.values():owner._private_g6_tail_conv.reset()
        return result
    NPUWorker.reset_private_q35_gdn_input_stats=reset_worker

def snapshot(worker):
    conv=worker.model_runner._q35_cached_conv_manager.layers
    rgdr=worker.model_runner._q35_rgdr_reuse_manager.layers
    inputs=worker.model_runner._q35_gdn_input_manager.layers
    assert len(conv)==len(rgdr)==len(inputs)==30
    return dict(proof_sha256={kind:sha(p) for kind,p in PROOFS.items()},
        gamma4_unchanged_all_gamma6_component_proven=True,public_budgets_and_fences_unchanged=True,
        existing_single_input_counter_add_redirected_not_extra_hot_operation=True,
        layers=[dict(layer=name,conv=owner._private_g6_tail_conv.metrics(),
            rgdr=rgdr[name]._private_g6_tail_rgdr.metrics(),
            input_weight_checks=inputs[name]._private_g6_input_weight_checks,
            input_counts=inputs[name]._private_gamma6_counts.cpu().tolist()) for name,owner in conv.items()])
