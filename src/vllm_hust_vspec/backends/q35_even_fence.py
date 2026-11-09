"""Read-only physical Mamba cache proof for one adjacent even-gamma change."""
import os
from ..adaptive.even_fence_contract import geometry

def prepare(runner,transition_id,old,new,stage,expected_roster):
    import torch
    if os.environ.get('HUST_VSPEC_EVEN_ROUND_ENABLE')!='1':
        raise RuntimeError('Even-round live switching is explicitly opt-in')
    if type(transition_id) is not int or transition_id<=0:
        raise ValueError('Positive transition ID required')
    g=geometry(old,new,stage)
    runner._vspec_even_pending_fence=None
    torch.npu.synchronize()
    n=runner.input_batch.num_reqs;roster=tuple(runner.input_batch.req_ids[:n])
    proof=dict(eligible=False,pid=os.getpid(),transition_id=transition_id,stage=stage,
               old_gamma=old,new_gamma=new,old_actual_width=old+1,new_actual_width=new+1,
               roster=list(roster),device_fenced=True,cache_tensors_rewritten=False,
               sampled_counts_rewritten=False,previous_draft_tensor_rewritten=False)
    def reject(reason):proof['reason']=reason;return proof
    if not 1<=n<=16 or len(set(roster))!=n or len(expected_roster)!=n or set(expected_roster)!=set(roster):
        return reject('worker_roster_not_exact_current_cohort')
    if (not runner.use_async_spec_decode or runner.use_dcp or runner.cache_config.mamba_cache_mode!='align'
            or runner.speculative_config.method!='mtp' or runner.speculative_config.num_speculative_tokens!=8
            or runner.parallel_config.tensor_parallel_size!=2
            or getattr(runner.parallel_config,'data_parallel_size',1)!=1
            or getattr(runner.parallel_config,'pipeline_parallel_size',1)!=1
            or runner.dtype!=torch.bfloat16 or runner.lora_config is not None
            or getattr(runner.model_config,'quantization',None) is not None
            or getattr(runner.model_config.hf_text_config,'model_type','') not in ('qwen3_5_moe','qwen3_5_moe_text')):
        raise RuntimeError('Even fence requires native BF16 Q35 TP2 PP1 DP1 async max8')
    if getattr(runner,'_vspec_mtp_synchronized_gamma',None)!=g['runtime']:
        return reject('runtime_width_mismatch')
    if stage=='wide' and getattr(runner,'_vspec_even_consumed_fence',None)!=(transition_id,'bridge'):
        return reject('matching_bridge_not_consumed')
    draft=runner._draft_token_ids
    if not isinstance(draft,torch.Tensor) or draft.shape!=(n,g['previous_draft']) or runner.prev_num_spec_tokens!=g['previous_draft']:
        return reject('previous_draft_tensor_or_stride_changed')
    counts=runner.valid_sampled_token_count_gpu
    if not isinstance(counts,torch.Tensor) or counts.numel()<n:
        return reject('missing_actual_device_accepted_counts')
    accepted=counts.flatten()[:n].cpu().tolist()
    if any(type(a) is not int or not 1<=a<=g['accepted_max'] for a in accepted):
        return reject('accepted_checkpoint_outside_preserved_prefix')
    refresh=getattr(runner,'_q35_fused_gdn_refresh',None)
    signatures=None if refresh is None else refresh.storage_signatures.get(old+1)
    if refresh is None or refresh.last_gamma!=old or not isinstance(signatures,tuple) or len(signatures)!=3:
        return reject('missing_actual_old_width_metadata_witness')
    if len(runner.attn_groups)!=4 or any(len(group)!=1 for group in runner.attn_groups):
        return reject('unexpected_attention_group_geometry')
    builders=[(gid,runner.attn_groups[gid][0].get_metadata_builder(0)) for gid in (1,2,3)]
    sizes=[b.kv_cache_spec.block_size for _,b in builders]
    if any(type(s) is not int or s<=0 for s in sizes) or len(set(sizes))!=1:
        return reject('inconsistent_actual_block_sizes')
    size=sizes[0];seq=runner.seq_lens[:n];lengths=seq.cpu().tolist()
    if any(x<=g['backward'] or (x-g['backward']-1)//size!=(x-1)//size
           or (x-1)//size!=(x+g['forward']-1)//size for x in lengths):
        return reject('near_bidirectional_mamba_block_boundary')
    groups=[]
    for index,(gid,builder) in enumerate(builders):
        previous=getattr(builder,'_vspec_adaptive_spec_state_indices_by_width',{}).get(old+1)
        if not isinstance(previous,torch.Tensor) or previous.ndim!=2 or previous.shape[1]!=old+1 or previous.shape[0]<n:
            return reject('missing_actual_old_width_buffer')
        signature=signatures[index]
        if signature[0]!=id(builder) or signature[1][0]!=(previous.data_ptr(),tuple(previous.shape),previous.dtype):
            return reject('actual_old_width_storage_signature_mismatch')
        table=runner.input_batch.block_table[gid].get_device_tensor(n)
        start=((seq-1)//size).long();width=max(old,new)+1
        if not bool(((start>=0)&(start+width<=table.shape[1])).all().cpu().item()):
            return reject('canonical_source_columns_out_of_bounds')
        columns=start[:,None]+torch.arange(width,dtype=torch.int64,device=seq.device)[None,:]
        canonical=table.gather(1,columns)
        if not torch.equal(canonical[:,:old+1].contiguous(),previous[:n]) or not bool((canonical>0).all().cpu().item()):
            return reject('actual_state_not_canonical_or_next_slots_unallocated')
        groups.append(dict(gid=gid,mamba_block_size=size,previous_actual_canonical=True,
                           accepted_state_retained=True,next_slots_allocated=True,actual_previous_width=old+1))
    layers=[m for _,m in runner.model.named_modules() if all(hasattr(m,k) for k in ('rearrange_mixed_qkv','A_log','dt_bias','head_k_dim'))]
    if len(layers)!=30:return reject('missing_thirty_native_GDN_layers')
    for layer in layers:
        conv,recurrent=layer.kv_cache
        if (conv.dtype!=torch.bfloat16 or tuple(conv.shape[-2:])!=(11,4096) or conv.stride(-1)!=1
                or recurrent.dtype!=torch.float32 or tuple(recurrent.shape[-3:])!=(16,128,128)):
            return reject('cache_geometry_not_max8_BF16_conv_FP32_state')
    proof.update(eligible=True,reason='ACTUAL_READ_ONLY_EVEN_PREFIX_ADMITTED',
                 previous_draft_width=g['previous_draft'],previous_draft_stride=g['previous_draft'],
                 mamba_block_size=size,accepted_counts=accepted,old_end_seq_lens=lengths,groups=groups,
                 all30_cache_layouts_validated=True,actual_width_cache_address_signature_validated=True)
    runner._vspec_even_pending_fence=proof
    return proof

def install():
    if os.environ.get('HUST_VSPEC_EVEN_ROUND_ENABLE')!='1':return False
    from vllm_ascend.worker.worker import NPUWorker
    prior=getattr(NPUWorker,'fence_private_even_mtp_transition',None)
    if prior is not None:
        if not getattr(prior,'_vspec_even_round_rpc',False):raise RuntimeError('Conflicting even transition RPC')
        return False
    def fence_private_even_mtp_transition(worker,*args):return prepare(worker.model_runner,*args)
    fence_private_even_mtp_transition._vspec_even_round_rpc=True
    NPUWorker.fence_private_even_mtp_transition=fence_private_even_mtp_transition
    return True
