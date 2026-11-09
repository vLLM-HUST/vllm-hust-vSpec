"""Gamma8 batch1..16 input fusion; component and all30 real-weight gates."""
import ast
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
from textwrap import dedent
from vllm_hust_vspec.backends.experimental_paths import artifact_dir

COMPONENT=artifact_dir('VSPEC_GAMMA8_INPUT_DIR')
PROOF=COMPONENT/'gamma8-input-fusion-c46-primitive-audit-r1.json'
ASCEND=artifact_dir('VSPEC_ASCEND_SOURCE_ROOT')/'vllm_ascend'
KERNEL=ASCEND/'ops/triton/q35_fused_gdn_inputs.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    p=json.loads(PROOF.read_text());assert p['completed'] and p['component_only'] and p['physical_npus']==[4,6]
    assert p['eligible_tokens']==[n*9 for n in range(1,17)] and p['only_gamma8_all16_batches']
    for part in (p,json.loads((COMPONENT/'launch.json').read_text())):
        assert all(sha(name)==digest for name,digest in part['sha256'].items())
    assert sha(KERNEL)==sha(COMPONENT/'kernel.py')
    assert {r['device'] for r in p['ranks']}=={0,1}
    for r in p['ranks']:
        assert r['passed'] and r['native_qk_norm_value_beta_bitexact']
        cases=[c for c in r['cases'] if 'phase' in c];assert len(cases)==128
        assert {c['tokens'] for c in cases}=={n*9 for n in range(1,17)}
        for c in cases:
            assert c['original_native_eager_self_graph_bits_exact']
            assert c['actual_consumed_BF16_bits_including_signed_zero_exact']
            assert c['fused_eager_self_graph_bits_exact'] and c['complete_all_five_inputs_raw_bits_readonly']
            assert c['canaries_unchanged'] and c['fp32_gate_atol']==c['fp32_gate_rtol']==1e-6
    return sha(PROOF)

def extend(original):
    source=inspect.getsource(original)
    old='tokens in (n * 5, n * 7)';new='(tokens in (n * 5, n * 7) or (1 <= n <= 16 and tokens == n * 9))'
    assert source.count(old)==1
    changed=source.replace(old,new);assert changed.replace(new,old)==source
    scope=dict(original.__globals__)
    exec(compile(ast.parse(changed),'<private-gamma8-input-domain>','exec'),scope)
    return scope[original.__name__]

def disambiguate_counter(original):
    # One existing add_ only: redirect gamma8 to an unambiguous per-request
    # cell. Equal token totals can otherwise mix gamma8/B5 and gamma4/B9.
    # The actual fused-input kernel and every numerical/fallback branch stay
    # unchanged; gamma4/6 still update the exact original counter cells.
    source=dedent(inspect.getsource(original))
    old='self.counts[tokens].add_(1)'
    new='(self._private_gamma8_counts[meta.num_spec_decodes] if tokens == meta.num_spec_decodes * 9 else self.counts[tokens]).add_(1)'
    assert source.count(old)==1
    changed=source.replace(old,new);assert changed.replace(new,old)==source
    scope=dict(original.__globals__)
    exec(compile(ast.parse(changed),'<private-gamma8-input-counter>','exec'),scope)
    return scope[original.__name__]

def real_weight_admission(owner):
    import torch
    from vllm_ascend.ops import gdn
    from vllm_ascend.ops.triton.q35_fused_gdn_inputs import fused_inputs
    attention=owner.attention;device=attention.A_log.device
    assert attention.A_log.numel()==attention.dt_bias.numel()==16
    weights=(attention.A_log,attention.dt_bias)
    original_weights=[w.clone() for w in weights];records=[]
    for requests,phase in ((n,p) for n in (1,14,16) for p in (0,1)):
        tokens=requests*9
        x=torch.arange(tokens*4096,dtype=torch.float32,device=device).reshape(tokens,4096)
        mixed=((x.remainder(257)-128)/64).to(torch.bfloat16)
        gates=torch.arange(tokens*16,dtype=torch.float32,device=device).reshape(tokens,16)
        a=((gates.remainder(63)-31)/8).to(torch.bfloat16)
        b=((gates.remainder(33)-16)/4).to(torch.bfloat16)
        if phase:
            mixed.zero_();mixed.view(torch.int16)[:,::2]=-32768
        inputs=(mixed,a,b,*weights);copies=[t.clone() for t in inputs]
        actual=fused_inputs(mixed,a,b,*weights)
        q,k,v=attention.rearrange_mixed_qkv(mixed)
        q,k=gdn.l2norm_fwd(q),gdn.l2norm_fwd(k)
        g,beta=gdn.DeviceOperator.fused_gdn_gating(attention.A_log,a,b,attention.dt_bias)
        native=(q,k,v,g,beta);torch.npu.synchronize()
        for index in (0,1,2,4):
            assert torch.equal(native[index].view(torch.int16),actual[index].view(torch.int16))
        torch.testing.assert_close(native[3],actual[3],rtol=1e-6,atol=1e-6)
        assert all(bool(torch.isfinite(t).all().cpu()) for t in actual)
        for left,right in zip(inputs,copies):
            dtype=torch.int16 if left.dtype==torch.bfloat16 else torch.int32
            assert torch.equal(left.view(dtype),right.view(dtype))
        records.append(dict(phase=phase,tokens=tokens,requests=requests,actual_model_A_log_dt_bias_used=True,
            qkv_beta_all_BF16_bits_including_signed_zero_exact=True,
            g_FP32_atol=1e-6,g_FP32_rtol=1e-6,complete_inputs_and_weights_bits_readonly=True))
    for left,right in zip(weights,original_weights):
        dtype=torch.int16 if left.dtype==torch.bfloat16 else torch.int32
        assert torch.equal(left.view(dtype),right.view(dtype))
    return records

def install():
    proof_sha=verify()
    from vllm_ascend.ops import gdn_input_fusion as native
    native.eligible=extend(native.eligible)
    original=native.GDNInputFusion.__init__
    original_reset=native.GDNInputFusion.reset_stats
    @wraps(original)
    def initialize(owner,*args,**kwargs):
        original(owner,*args,**kwargs)
        assert owner.enabled and owner.mode in ('on','validate')
        import torch
        owner._private_gamma8_counts=torch.zeros(17,dtype=torch.int64,device=owner.attention.A_log.device)
        owner._private_gamma8_input_weight_checks=real_weight_admission(owner)
    @wraps(original_reset)
    def reset(owner):
        original_reset(owner)
        if owner.enabled:owner._private_gamma8_counts.zero_()
    native.GDNInputFusion.__init__=initialize
    native.GDNInputFusion.reset_stats=reset
    native.GDNInputFusion.apply=disambiguate_counter(native.GDNInputFusion.apply)
    native._private_gamma8_input_receipt=dict(proof_sha256=proof_sha,kernel_sha256=sha(KERNEL),
        only_width_eligibility_extended_to_gamma8_batches1_to16=True,
        original_apply_math_and_physical_fence_sources_unchanged=True,
        gamma8_existing_single_counter_add_redirected_NOT_extra_hot_operation=True,
        gamma4_gamma6_original_counter_cells_unchanged=True,
        g_FP32_tolerance_same_as_existing_gamma4_gamma6=True,
        NOT_whole_model_unfused_native_full_state_bitexact_claim=True)

def snapshot(worker):
    from vllm_ascend.ops import gdn_input_fusion as native
    assert native.eligible.__code__.co_filename=='<private-gamma8-input-domain>'
    receipt=dict(native._private_gamma8_input_receipt)
    assert receipt['proof_sha256']==sha(PROOF) and receipt['kernel_sha256']==sha(KERNEL)
    layers=worker.model_runner._q35_gdn_input_manager.layers
    assert len(layers)==30
    receipt['actual_weight_checks']=[dict(layer=name,cases=owner._private_gamma8_input_weight_checks)
        for name,owner in layers.items()]
    receipt['actual_gamma8_input_replays']=[dict(layer=name,
        counts_by_requests=owner._private_gamma8_counts.cpu().tolist()) for name,owner in layers.items()]
    return receipt
