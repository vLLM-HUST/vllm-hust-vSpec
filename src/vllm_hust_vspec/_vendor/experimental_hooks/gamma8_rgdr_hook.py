"""Unselected private gamma8 extension; refuses missing independent NPU proof."""
import hashlib
import json
from pathlib import Path
from functools import wraps

from vllm_hust_vspec.backends.experimental_paths import artifact_dir, require_current_npu_proof

COMPONENT=artifact_dir('VSPEC_GAMMA8_RGDR_DIR')
PROOF=COMPONENT/'runs/gamma8-allb-c46-r1.primitive-audit.json'
LIBRARY=COMPONENT/'build-r1/lib/libvspec_rgdr_edge8_adapter.so'

def verify():
    proof=json.loads(PROOF.read_text())
    require_current_npu_proof(proof)
    assert proof['passed'] and proof['exact']
    assert {r['rank'] for r in proof['ranks']}=={0,1}
    for part in (proof,*proof['ranks']):
        for filename,digest in part['sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    for rank in proof['ranks']:
        assert len(rank['records'])==192
        assert {(c['requests'],c['bank'],c['seed']) for c in rank['records']}=={
            (n,b,s) for n in range(1,17) for b in (192,647) for s in range(6)}
        for case in rank['records']:
            assert case['independent_native_eager_graph_reference']
            assert case['eight_eager_graph_revisions_bits_exact']
            assert case['captured_graph_dynamic_lengths_and_accepted_covered']
            assert case['full_FP32_state_including_untouched_NaN_payload_and_signed_zero_exact']
            assert case['complete_inputs_bits_readonly'] and case['entire_folded_counter_array_exact']
    return hashlib.sha256(PROOF.read_bytes()).hexdigest()

def eligible(torch,meta,inputs,state,lengths,indices,accepted,fused):
    q,k,v,g,beta=inputs
    n=meta.num_spec_decodes;tokens=n*9
    return bool(fused and not meta.num_prefills and not meta.num_decodes
        and 1<=n<=16 and meta.spec_sequence_masks is not None
        and q.shape==k.shape==(tokens,8,128) and v.shape==(tokens,16,128)
        and g.shape==beta.shape==(tokens,16)
        and state.ndim==4 and state.shape[0]>=tokens and state.shape[1:]==(16,128,128)
        and lengths.shape==(n+1,) and accepted.shape==(n,) and indices.shape==(tokens,)
        and q.dtype==k.dtype==v.dtype==beta.dtype==torch.bfloat16
        and g.dtype==state.dtype==torch.float32
        and lengths.dtype==accepted.dtype==indices.dtype==torch.int32
        and all(t.is_contiguous() and t.device==q.device
            for t in (*inputs,state,lengths,indices,accepted)))

class Gamma8RGDR:
    def __init__(self,torch,owner,proof_sha):
        self.torch,self.owner,self.proof_sha=torch,owner,proof_sha
        self.counts=torch.zeros(17,dtype=torch.int32,device=owner.attention.A_log.device)
        self.errors=torch.zeros(2,dtype=torch.int32,device=owner.attention.A_log.device)
        self.captured=False

    def apply(self,inputs,state,lengths,indices,accepted):
        torch=self.torch;q,k,v,g,beta=inputs;owner=self.owner
        if owner.mode=='validate':
            reference_state,differences=owner.workspace.acquire(state)
            reference_state.copy_(state)
        result=torch.ops.vspec_rgdr_edge8.project(q,k,v,beta,state,lengths,indices,g,accepted,self.counts,128,1)
        if owner.mode=='validate':
            reference=torch.ops._C_ascend.npu_recurrent_gated_delta_rule(query=q,key=k,value=v,
                state=reference_state,beta=beta,g=g,scale=128**-.5,actual_seq_lengths=lengths,
                ssm_state_indices=indices,num_accepted_tokens=accepted)
            torch.ne(state.view(torch.int32),reference_state.view(torch.int32),out=differences)
            flags=torch.stack(((result.view(torch.int16)!=reference.view(torch.int16)).any(),
                differences.any())).to(torch.int32)
            torch.maximum(self.errors,flags,out=self.errors)
        self.captured=True
        return result

    def reset(self):self.counts.zero_();self.errors.zero_()

    def metrics(self):
        return dict(mode=self.owner.mode,captured=self.captured,
            v_step=128,output_buffers=1,
            actual_same_kernel_folded_replay_counter_cells=self.counts.cpu().tolist(),
            validation_error_flags=self.errors.cpu().tolist(),
            ON_is_component_proven_NOT_per_frame_runtime_native_shadow=self.owner.mode=='on',
            independent_component_proof_sha256=self.proof_sha,
            original_gamma4_and6_apply_fallback_unchanged=True,
            state_dtype='torch.float32',input_dtype='torch.bfloat16',stats_read_after_timers_only=True)

def install():
    proof_sha=verify()  # MUST precede library loading or any model construction.
    import torch
    from vllm_ascend.ops.q35_rgdr_reuse import RGDRReuse
    assert not getattr(RGDRReuse.apply,'_private_gamma8_rgdr',False)
    torch.ops.load_library(str(LIBRARY))
    def fake(query,key,value,beta,state,lengths,indices,g,accepted,counters,v_step,output_buffers):
        return torch.empty_like(value)
    torch.library.register_fake('vspec_rgdr_edge8::project')(fake)
    original_init,original_apply,original_reset=RGDRReuse.__init__,RGDRReuse.apply,RGDRReuse.reset
    @wraps(original_init)
    def initialize(owner,*args,**kwargs):
        original_init(owner,*args,**kwargs)
        owner._private_gamma8_rgdr=Gamma8RGDR(torch,owner,proof_sha)
    @wraps(original_apply)
    def apply(owner,meta,inputs,state,lengths,indices,accepted,fused):
        if eligible(torch,meta,inputs,state,lengths,indices,accepted,fused):
            return owner._private_gamma8_rgdr.apply(inputs,state,lengths,indices,accepted)
        return original_apply(owner,meta,inputs,state,lengths,indices,accepted,fused)
    @wraps(original_reset)
    def reset(owner):
        result=original_reset(owner);owner._private_gamma8_rgdr.reset();return result
    apply._private_gamma8_rgdr=True
    RGDRReuse.__init__,RGDRReuse.apply,RGDRReuse.reset=initialize,apply,reset
