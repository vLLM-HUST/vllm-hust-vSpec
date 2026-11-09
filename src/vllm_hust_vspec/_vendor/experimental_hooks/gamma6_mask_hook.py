"""Target physical-gamma6 native-top8 CONSUMER fusion, not a router change.

Native routing/quant-method/dispatch bodies are untouched. A scoped CPU-only
owner tag binds the already selected experts' map to its own witness tensor.
"""
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from vllm_hust_vspec.backends.experimental_paths import artifact_dir

COMPONENT=artifact_dir('VSPEC_GAMMA6_MASK_DIR')
PROOF=COMPONENT/'primitive-audit.json'
SCOPE='_private_g6_post_top8_owner'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def version_of(tensor):
    return None if tensor.is_inference() else tensor._version

def verify():
    proof=read(PROOF)
    assert proof['passed'] and proof['exact'] and proof['physical_npus']==[4,6]
    assert proof['only_post_native_top8_consumer_no_router_change']
    assert {rank['rank'] for rank in proof['ranks']}=={0,1}
    for rank in proof['ranks']:
        assert rank['passed'] and len(rank['records'])==64
        assert {(c['requests'],c['id_dtype'],c['weight_dtype']) for c in rank['records']}=={
            (n,i,w) for n in range(1,17) for i in ('torch.int32','torch.int64') for w in ('torch.bfloat16','torch.float32')}
        for case in rank['records']:
            assert all(case[key] for key in ('independent_native_eager_self_graph_bits_exact',
                'eight_dynamic_input_eager_graph_revisions_bits_exact',
                'all_raw_output_bits_NaN_payload_signed_zero_infinity_subnormal_covered',
                'complete_input_storage_including_canaries_readonly','all17_counter_cells_exact',
                'router_scores_top8_and_weight_values_unchanged'))
            # Every ABI/batch must improve over the unchanged three-op native
            # consumer, not just a selected friendly case.
            assert case['median_us']['native']/case['median_us']['fused_folded_count']>=1.01
    for part in (proof,*proof['ranks']):
        assert all(sha(path)==digest for path,digest in part['sha256'].items())
    return sha(PROOF)

def geometry(descriptor,is_draft,is_draft_prefill,rows):
    if (is_draft or is_draft_prefill or descriptor is None or not getattr(descriptor,'uniform',False)
            or type(descriptor.num_reqs) is not int or not 1<=descriptor.num_reqs<=16
            or descriptor.num_tokens!=rows or rows!=descriptor.num_reqs*7):
        return None
    return descriptor.num_reqs

def call_with_owner(context,layer,invoke):
    missing=object();previous=getattr(context,SCOPE,missing)
    setattr(context,SCOPE,layer)
    try:return invoke()
    finally:
        if previous is missing:delattr(context,SCOPE)
        else:setattr(context,SCOPE,previous)

def kernel_module():
    name='_private_proven_g6_mask'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,COMPONENT/'kernel.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name]
    assert Path(module.__file__).resolve()==COMPONENT/'kernel.py'
    return module

def check_real_map(torch,expert_map):
    assert expert_map.shape==(256,) and expert_map.dtype==torch.int32
    assert expert_map.device.type=='npu' and expert_map.is_contiguous()
    records=[];version=version_of(expert_map)
    for n,phase in ((n,p) for n in range(1,17) for p in (0,1)):
        ids=((torch.arange(n*7*8,device=expert_map.device)+phase*19)%256).to(torch.int32).reshape(n*7,8)
        weights=((torch.arange(n*7*8,device=expert_map.device).remainder(31)-15)/32).to(torch.bfloat16).reshape_as(ids)
        counts=torch.zeros(17,dtype=torch.int32,device=expert_map.device)
        saved=[value.clone() for value in (ids,weights,expert_map)]
        expected=weights*(expert_map[ids]!=-1)
        actual=torch.ops.vspec_g6_mask.project(ids,weights,expert_map,counts)
        torch.npu.synchronize()
        assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
        assert all(torch.equal(value.view(torch.int16) if value.dtype==torch.bfloat16 else value,
            before.view(torch.int16) if before.dtype==torch.bfloat16 else before)
            for value,before in zip((ids,weights,expert_map),saved))
        wanted=torch.zeros_like(counts);wanted[n]=1;assert torch.equal(counts,wanted)
        assert version_of(expert_map)==version
        records.append(dict(requests=n,phase=phase,actual_original_expert_map_used=True,
            full_native_BF16_masked_weight_bits_exact=True,all_inputs_readonly=True,all17_counter_cells_exact=True))
    return records

def snapshot(worker):
    rows=[]
    for name,layer in worker.model_runner._private_g6_post8_layers:
        entry=layer._private_g6_post8
        assert layer.ascend_expert_map is entry['map'] and version_of(entry['map'])==entry['version']
        rows.append(dict(layer=name,counts=entry['counts'].cpu().tolist(),
            actual_map_shape=list(entry['map'].shape),actual_map_dtype=str(entry['map'].dtype),
            actual_map_device=str(entry['map'].device),actual_map_version=entry['version'],
            original_map_values=entry['map'].cpu().tolist(),real_map_checks=entry['checks']))
    assert len(rows)==40
    return dict(proof_sha256=sha(PROOF),mode='on',layers=rows,only_target_physical_gamma6=True,
        gamma2_gamma4_gamma8_and_draft_fallback_unchanged=True,
        native_router_scores_selected_experts_and_weights_unchanged=True,
        native_quant_method_and_dispatch_bodies_and_all_GPU_fences_unchanged=True,
        new_scoped_CPU_only_owner_tag_disclosed=True,
        counts_are_same_kernel_NOT_Python_call_or_capture_only=True,
        ON_component_proven_NOT_per_frame_full_model_native_shadow=True,
        stats_read_after_timers_only=True)

def install():
    proof_hash=verify() # Before registration/model construction, never an after-the-fact gate.
    import torch
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.ops.fused_moe.q35_expert_mask import Q35ExpertMask
    from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod,AscendRoutedExperts
    from vllm_ascend.worker.worker import NPUWorker
    assert not getattr(Q35ExpertMask.apply,'_private_g6_post_top8',False)
    kernel=kernel_module()
    assert kernel._mask.cache_key=='a48d3da660c532554b547bcbecdd44271d058c58d39e350c479dea33952a0cd2'
    definition=torch.library.Library('vspec_g6_mask','DEF')
    definition.define('project(Tensor ids, Tensor weights, Tensor expert_map, Tensor(a!) counts) -> Tensor')
    implementation=torch.library.Library('vspec_g6_mask','IMPL','PrivateUse1')
    implementation.impl('project',kernel.fused)
    @torch.library.register_fake('vspec_g6_mask::project')
    def fake(ids,weights,expert_map,counts):return torch.empty_like(weights)
    # Lifetime owned by the existing helper class, no module-global tensor map.
    Q35ExpertMask._private_g6_post8_registration=(definition,implementation)
    original_quant=AscendUnquantizedFusedMoEMethod.apply
    original_mask=Q35ExpertMask.apply
    original_load=NPUWorker.load_model
    original_reset=NPUWorker.reset_private_q35_gdn_input_stats

    @wraps(original_mask)
    def apply_mask(owner,ids,weights,expert_map):
        if ids.ndim!=2:return original_mask(owner,ids,weights,expert_map)
        try:context=get_forward_context()
        except AssertionError:return original_mask(owner,ids,weights,expert_map)
        layer=getattr(context,SCOPE,None)
        entry=getattr(layer,'_private_g6_post8',None) if layer is not None else None
        n=geometry(getattr(context,'batch_descriptor',None),_EXTRA_CTX.is_draft_model,
            _EXTRA_CTX.is_draft_model_prefill,ids.shape[0])
        if (entry is None or n is None or owner.enabled or owner.mode!='off'
                or entry['map'] is not expert_map or version_of(expert_map)!=entry['version']
                or ids.ndim!=2 or ids.shape!=(n*7,8) or weights.shape!=ids.shape
                or ids.dtype not in (torch.int32,torch.int64)
                or weights.dtype not in (torch.bfloat16,torch.float32)
                or expert_map.shape!=(256,) or expert_map.dtype!=torch.int32
                or not all(value.is_contiguous() and value.device==weights.device for value in (ids,weights,expert_map))):
            return original_mask(owner,ids,weights,expert_map)
        return torch.ops.vspec_g6_mask.project(ids,weights,expert_map,entry['counts'])

    @wraps(original_quant)
    def apply_quant(method,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        context=get_forward_context()
        n=geometry(getattr(context,'batch_descriptor',None),_EXTRA_CTX.is_draft_model,
            _EXTRA_CTX.is_draft_model_prefill,topk_ids.shape[0])
        def invoke():
            return original_quant(method,layer=layer,x=x,topk_weights=topk_weights,topk_ids=topk_ids,
                shared_experts=shared_experts,shared_experts_input=shared_experts_input)
        if n is None or not hasattr(layer,'_private_g6_post8'):return invoke()
        return call_with_owner(context,layer,invoke)

    @wraps(original_load)
    def load_model(worker,*args,**kwargs):
        result=original_load(worker,*args,**kwargs)
        config=worker.vllm_config
        assert config.model_config.dtype==torch.bfloat16 and config.model_config.quantization is None
        assert config.parallel_config.tensor_parallel_size==2 and config.parallel_config.data_parallel_size==1
        assert not config.lora_config
        layers=[(name,module) for name,module in worker.model_runner.model.named_modules()
            if isinstance(module,AscendRoutedExperts)]
        assert len(layers)==40 and all(not layer.dynamic_eplb for name,layer in layers)
        for name,layer in layers:
            expert_map=layer.ascend_expert_map
            assert not hasattr(layer,'_private_g6_post8')
            with torch.inference_mode(False):counts=torch.zeros(17,dtype=torch.int32,device=layer.w13_weight.device)
            assert not counts.is_inference()
            layer._private_g6_post8=dict(counts=counts,map=expert_map,version=version_of(expert_map),
                checks=check_real_map(torch,expert_map))
        worker.model_runner._private_g6_post8_layers=tuple(layers)
        print('[private-g6-post-top8-load] '+json.dumps(dict(proof_sha256=proof_hash,layers=40,
            all40_real_original_maps_independently_checked=True,only_target_width7=True,
            router_scores_experts_weights_and_all_native_fences_unchanged=True)),flush=True)
        return result

    @wraps(original_reset)
    def reset(worker):
        result=original_reset(worker)
        for name,layer in worker.model_runner._private_g6_post8_layers:layer._private_g6_post8['counts'].zero_()
        return result

    apply_mask._private_g6_post_top8=True
    Q35ExpertMask.apply=apply_mask
    AscendUnquantizedFusedMoEMethod.apply=apply_quant
    NPUWorker.load_model=load_model
    NPUWorker.reset_private_q35_gdn_input_stats=reset
