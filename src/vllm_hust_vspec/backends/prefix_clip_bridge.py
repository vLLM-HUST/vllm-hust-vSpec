"""Bounded verified-prefix bridge BEFORE the unchanged native count producer.

No existing accepted count, KV/SSM tensor or previous draft is an operand.
The existing physical shrink admission/consume implementation is untouched.
"""
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .experimental_paths import artifact_dir

COMPONENT=artifact_dir('VSPEC_PREFIX_CLIP_BRIDGE_DIR')
PROOF=COMPONENT/'prefix-bridge-component-audit-r1.json'

def verify():
    p=json.loads(PROOF.read_text())
    assert p['passed'] and p['physical_npus']==[4,6]
    assert p['component_only_NOT_live_async_bridge_state_or_quality_or_speed_admission']
    for part in (p,*p['ranks']):
        for filename,digest in part['sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    assert {r['device'] for r in p['ranks']}=={0,1}
    for rank in p['ranks']:
        assert len(rank['records'])==96
        assert {(r['requests'],r['old'],r['new'],r['discarded_requests']) for r in rank['records']}=={
            (n,g,g-2,s) for n in range(1,17) for g in (4,6,8) for s in (0,1)}
        for r in rank['records']:
            for k in ('nine_revisions_including_all_rejection_positions_and_all_accept',
                      'independent_native_wide_and_narrow_eager_self_graph',
                      'all_output_ID_bits_and_native_count_next_ID_bits_exact',
                      'original_wide_output_and_all_inputs_readonly',
                      'complete_in_kernel_folded_witness_exact',
                      'asynchronous_eager_chain_32calls_without_intermediate_sync_verified',
                      'native_reference_CU_guarded_backing_unselected_offset_minus1_read_explicit',
                      'no_attention_KV_SSM_or_existing_count_tensor_access'):
                assert r[k]
    return hashlib.sha256(PROOF.read_bytes()).hexdigest()

def valid_ticket(sequence,old,new,roster):
    return bool(type(sequence) is int and sequence>0 and old in (4,6,8) and new==old-2
        and 1<=len(roster)<=16 and len(set(roster))==len(roster))

class ClipOwner:
    def __init__(self,runner,torch,clip,proof_sha):
        self.runner,self.torch,self.clip,self.proof_sha=runner,torch,clip,proof_sha
        self.plan=None;self.arms=self.disarms=self.consumed=0;self.calls=0
        self.counts=torch.zeros(9,dtype=torch.int32,device=runner.device)
        self.preflight=[]
        for old in (4,6,8):
            source=torch.arange(14*(old+1),dtype=torch.int32,device=runner.device).reshape(14,old+1)
            saved=source.clone();out=clip(source,old,old-2,self.counts)
            expected=torch.full_like(source,-1);expected[:,:old-1]=source[:,:old-1]
            torch.npu.synchronize()
            assert torch.equal(source,saved) and torch.equal(out,expected)
            self.preflight.append(dict(old=old,new=old-2,source_readonly_and_prefix_bits_exact=True))
        self.counts.zero_();torch.npu.synchronize()

    def snapshot(self):
        return dict(enabled=True,component_proof_sha256=self.proof_sha,
            component_only_NOT_full_model_native_shadow=True,
            actual_folded_clip_counts_by_old_gamma=self.counts.cpu().tolist(),
            Python_forward_calls=self.calls,armed=self.arms,disarmed=self.disarms,
            consumed_matching_physical_shrink=self.consumed,active_plan=self.plan,
            original_count_producer_and_physical_fences_unchanged=True,
            no_KV_SSM_or_existing_accepted_count_operands=True,
            original_full_verified_output_not_mutated=True,preflight=self.preflight,
            all_clip_and_fence_costs_in_measured_wall=True,stats_read_after_timers_only=True)

def install():
    proof_sha=verify()
    import torch
    from dataclasses import replace
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker
    if getattr(AscendRejectionSampler.forward,'_private_prefix_clip_bridge',False):return False
    s=importlib.util.spec_from_file_location('private_verified_prefix_kernel',COMPONENT/'clip_kernel.py')
    module=importlib.util.module_from_spec(s);sys.modules[s.name]=module;s.loader.exec_module(module)
    clip=module.clip_verified_prefix
    original_load=NPUModelRunner.load_model
    @wraps(original_load)
    def load_model(runner,*args,**kwargs):
        result=original_load(runner,*args,**kwargs)
        runner._private_prefix_clip=ClipOwner(runner,torch,clip,proof_sha)
        runner.rejection_sampler._private_prefix_clip=runner._private_prefix_clip
        return result
    original_forward=AscendRejectionSampler.forward
    @wraps(original_forward)
    def forward(sampler,metadata,draft_probs,logits,sampling_metadata):
        output=original_forward(sampler,metadata,draft_probs,logits,sampling_metadata)
        owner=getattr(sampler,'_private_prefix_clip',None)
        if owner is None or owner.plan is None:return output
        plan=owner.plan;r=owner.runner;old,new=plan['old'],plan['new']
        actual=getattr(r,'_vspec_mtp_synchronized_gamma',None)
        if actual==new and getattr(r,'_vspec_even_consumed_fence',None)==(plan['sequence'],'shrink'):
            owner.plan=None;owner.consumed+=1
            return output
        if actual!=old:raise RuntimeError('Clip bridge lost matching actual width/consumed shrink witness')
        config=get_ascend_config().rejection_sampler_config
        if (not sampling_metadata.all_greedy or sampling_metadata.max_num_logprobs is not None
                or sampler.synthetic_mode or config.enable_block_verify or config.enable_entropy_verify):
            raise RuntimeError('Verified prefix bridge requires original plain greedy no-logprobs sampler')
        ids=output.sampled_token_ids
        roster=tuple(r.input_batch.req_ids[:r.input_batch.num_reqs])
        if (metadata.max_spec_len!=old or not metadata.num_draft_tokens
                or any(g!=old for g in metadata.num_draft_tokens)
                or not set(roster).issubset(plan['roster']) or ids.shape!=(len(roster),old+1)):
            raise RuntimeError('Clip bridge requires original uniform old-width native sampler output')
        clipped=owner.clip(ids,old,new,owner.counts)
        owner.calls+=1
        # This returns before any outer _sample/early-count wrapper sees IDs.
        return replace(output,sampled_token_ids=clipped)
    forward._private_prefix_clip_bridge=True
    NPUModelRunner.load_model=load_model;AscendRejectionSampler.forward=forward

    def arm(worker,sequence,old,new,roster):
        if not valid_ticket(sequence,old,new,roster):raise ValueError('Invalid owned adjacent shrink ticket')
        r=worker.model_runner;owner=r._private_prefix_clip
        torch.npu.synchronize()
        actual=tuple(r.input_batch.req_ids[:r.input_batch.num_reqs])
        if (getattr(r,'_vspec_mtp_synchronized_gamma',None)!=old
                or not set(roster).issubset(actual) or owner.plan is not None):
            raise RuntimeError('Clip arm lost live old-width roster or has an uncleared plan')
        owner.plan=dict(sequence=sequence,old=old,new=new,roster=tuple(roster))
        owner.arms+=1
        return dict(pid=__import__('os').getpid(),sequence=sequence,old=old,new=new,
            roster=list(roster),armed=True,device_fenced=True,
            no_cache_or_existing_count_mutation=True)
    def disarm(worker,sequence):
        owner=worker.model_runner._private_prefix_clip;torch.npu.synchronize()
        if owner.plan is not None and owner.plan['sequence']!=sequence:
            raise RuntimeError('Cannot disarm an unrelated clip ticket')
        owner.plan=None;owner.disarms+=1
        return dict(pid=__import__('os').getpid(),sequence=sequence,disarmed=True,device_fenced=True)
    NPUWorker.arm_private_prefix_clip_bridge=arm
    NPUWorker.disarm_private_prefix_clip_bridge=disarm
    return True
