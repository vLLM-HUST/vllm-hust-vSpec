"""Default-off exact draft postprojection vote; full target path untouched.

F.linear, BF16 local max, full target verifier, policy and collectives retain
their original math. Only proposal postprojection packing/unpacking and the
native all-gather layout change. Validation shadows the actual old NPU vote.
"""
from functools import wraps
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from .experimental_paths import artifact_dir

PRIMITIVE=artifact_dir('VSPEC_FUSED_PROPOSAL_VOTE_DIR')
PROOF=PRIMITIVE/'vote-r1-primitive-audit.json'
_LIBRARIES=[]


def verify_primitive():
    data=json.loads(PROOF.read_text())
    assert data['passed'] and data['component_only'] and data['cpu_tests']==5
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in data['sha256'].items())
    records=data['both_card_records'];assert {r['rank'] for r in records}=={0,1}
    for rank in records:
        assert rank['passed'] and rank['both_BF16_and_FP32_with_all49_special_score_pairs']
        assert len(rank['records'])==26
        assert all(len(r['checks'])==61 and not any(c['joint_bad'] for c in r['checks']) for r in rank['records'])
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in rank['sha256'].items())
    return data


def register_ops():
    if _LIBRARIES:return
    name='_vspec_proved_fused_proposal_vote_kernel'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PRIMITIVE/'kernel.py')
        kernel=importlib.util.module_from_spec(spec);sys.modules[name]=kernel;spec.loader.exec_module(kernel)
    kernel=sys.modules[name]
    definition=torch.library.Library('vspec_proposal_vote','DEF')
    definition.define('pack(Tensor values, Tensor indices, Tensor ids) -> Tensor')
    definition.define('unpack(Tensor gathered, Tensor(a!) counter) -> Tensor')
    impl=torch.library.Library('vspec_proposal_vote','IMPL','PrivateUse1')
    impl.impl('pack',kernel.pack);impl.impl('unpack',kernel.unpack)
    @torch.library.register_fake('vspec_proposal_vote::pack')
    def pack_fake(values,indices,ids):
        return torch.empty((values.numel(),2),device=values.device,dtype=torch.float32)
    @torch.library.register_fake('vspec_proposal_vote::unpack')
    def unpack_fake(gathered,counter):
        return torch.empty((gathered.shape[0]//2,),device=gathered.device,dtype=torch.int64)
    _LIBRARIES.extend((definition,impl))


def original_vote(compact,values,indices,n):
    ids=compact.ids[indices]
    packed=torch.stack((values.float(),ids.float()),dim=-1)
    gathered=compact.group.all_gather(packed,dim=-1).reshape(n,compact.group.world_size,2)
    winner=gathered[:,:,0].argmax(-1,keepdim=True)
    return gathered[:,:,1].gather(1,winner).squeeze(1).to(torch.int64)


def shadow_vote(compact,values,indices,tokens):
    n=values.numel()
    expected=original_vote(compact,values,indices,n)
    flag=(tokens!=expected).any().to(torch.int32)
    compact._q35_vote_checks[n].add_(1)
    torch.maximum(compact._q35_vote_errors[n],flag,out=compact._q35_vote_errors[n])


def project(self,hidden):
    logits = F.linear(hidden, self.weight, self.bias)
    if self.soft_cap is not None:
        logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
    if self.scale != 1:
        logits = logits * self.scale
    values, indices = logits.max(dim=-1)
    packed=torch.ops.vspec_proposal_vote.pack(values,indices,self.ids)
    gathered=self.group.all_gather(packed,dim=0)
    tokens=torch.ops.vspec_proposal_vote.unpack(gathered,self._q35_vote_counter)
    if self._q35_vote_validate:
        shadow_vote(self,values,indices,tokens)
    return tokens


project._q35_fused_proposal_vote=True


def validate_head(head,mode):
    from .mtp_active_head import BalancedCompactProposalHead
    if (type(head) is not BalancedCompactProposalHead or mode not in ('on','validate')
            or head.weight.shape!=(49152,2048) or head.weight.dtype!=torch.bfloat16
            or head.bias is not None or head.scale!=1 or head.soft_cap is not None
            or head.group.world_size!=2 or head.group.rank_in_group not in (0,1)
            or head.ids.shape!=(49152,) or head.ids.dtype!=torch.int64
            or not getattr(type(head).project,'_q35_fused_proposal_vote',False)):
        raise RuntimeError('Exact fused vote requires unchanged BF16 balanced96K TP2 proposal head')


def install_fused_proposal_vote():
    from ..config import q35_fused_proposal_vote_enabled,q35_owner_mid_twin_enabled,q35_inplace_head_nz_enabled
    if not q35_fused_proposal_vote_enabled():return False
    if not q35_owner_mid_twin_enabled() or not q35_inplace_head_nz_enabled():
        raise RuntimeError('Fused proposal vote requires independently proved owner/mid and inplace head lane')
    verify_primitive()
    from vllm_ascend import envs
    from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
    from vllm_ascend.worker.worker import NPUWorker
    from .mtp_active_head import CompactProposalHead,BalancedCompactProposalHead
    if envs.VLLM_ASCEND_Q35_GDN_INPUT_FUSION not in ('on','validate'):
        raise RuntimeError('Fused vote requires original explicit on/validate runtime')
    installed=getattr(AscendEagleProposer.load_model,'_q35_fused_proposal_vote',False)
    class_installed=getattr(BalancedCompactProposalHead.project,'_q35_fused_proposal_vote',False)
    if installed or class_installed:
        if installed and class_installed:return False
        raise RuntimeError('Partial fused-vote installation conflict')
    if BalancedCompactProposalHead.project is not CompactProposalHead.project:
        raise RuntimeError('Unknown proposal class projection hook')
    register_ops()
    BalancedCompactProposalHead.project=project
    original_load=AscendEagleProposer.load_model
    @wraps(original_load)
    def load_model(self,*args,**kwargs):
        result=original_load(self,*args,**kwargs)
        head=self._vspec_active_proposal_head
        mode=envs.VLLM_ASCEND_Q35_GDN_INPUT_FUSION
        validate_head(head,mode)
        if hasattr(head,'_q35_vote_counter'):raise RuntimeError('Duplicate fused-vote allocation')
        head._q35_vote_validate=mode=='validate'
        # Ordinary mutable tensors: inference tensors are illegal when a graph
        # replays outside the creation inference context.
        with torch.inference_mode(False):
            head._q35_vote_counter=torch.zeros(8193,device=head.weight.device,dtype=torch.int32)
            if head._q35_vote_validate:
                head._q35_vote_checks=torch.zeros(8193,device=head.weight.device,dtype=torch.int64)
                head._q35_vote_errors=torch.zeros(8193,device=head.weight.device,dtype=torch.int32)
        assert not head._q35_vote_counter.is_inference()
        receipt=dict(pid=os.getpid(),rank=head.group.rank_in_group,mode=mode,
            original_linear_and_BF16_local_max_unchanged=True,full_target_projection_and_verifier_unchanged=True,
            native_same_TP_group_all_gather_rank_major_layout=True,actual_NPU_positive_zero_precedence_preserved=True,
            ON_no_instance_project_hook_or_ND_copy=mode=='on',all_device_stats_read_after_timers_only=True,
            source_sha256=hashlib.sha256(Path(inspect.getsourcefile(project)).read_bytes()).hexdigest())
        head._q35_vote_receipt=receipt
        print('[q35-fused-proposal-vote-load] '+json.dumps(receipt),flush=True)
        return result
    load_model._q35_fused_proposal_vote=True
    AscendEagleProposer.load_model=load_model
    original_metrics=NPUWorker.get_private_q35_gdn_input_metrics
    original_reset=NPUWorker.reset_private_q35_gdn_input_stats
    @wraps(original_metrics)
    def metrics(self):
        rows=original_metrics(self);head=self.model_runner.drafter._vspec_active_proposal_head
        counts=head._q35_vote_counter.cpu().tolist()
        checks=head._q35_vote_checks.cpu().tolist() if head._q35_vote_validate else None
        errors=head._q35_vote_errors.cpu().tolist() if head._q35_vote_validate else None
        assert rows and getattr(type(head).project,'_q35_fused_proposal_vote',False)
        rows[0]['fused_proposal_vote']=dict(head._q35_vote_receipt,
            actual_replay_counts={str(i):v for i,v in enumerate(counts) if v},
            native_vote_shadow_counts={str(i):v for i,v in enumerate(checks) if v} if checks is not None else None,
            native_vote_shadow_error_flags=errors)
        return rows
    @wraps(original_reset)
    def reset(self):
        result=original_reset(self);head=self.model_runner.drafter._vspec_active_proposal_head
        head._q35_vote_counter.zero_()
        if head._q35_vote_validate:head._q35_vote_checks.zero_();head._q35_vote_errors.zero_()
        return result
    NPUWorker.get_private_q35_gdn_input_metrics=metrics
    NPUWorker.reset_private_q35_gdn_input_stats=reset
    return True
