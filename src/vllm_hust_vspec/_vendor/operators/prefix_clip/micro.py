"""Native greedy old/new-width reference plus untouched native GPU count tail.

No model, attention, KV or SSM state is passed to this component. A pass is
NOT a proof that a live async bridge or its physical fences are integrated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as Obj
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    import torch_npu
    from clip_kernel import clip_verified_prefix
    from native_count_tail import load,SOURCE
    from vllm_ascend.ops.triton import reject_sample as native_sample_module
    from vllm_ascend.ops.triton.reject_sample import rejection_greedy_sample_with_triton,cal_grid_and_block_size
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    os.umask(0o077);torch.npu.set_device(a.device);torch.set_num_threads(1)
    init_device_properties_triton();count_tail=load();records=[]
    for n,old,seed in ((n,g,s) for n in range(1,17) for g in (4,6,8) for s in (0,1)):
        new=old-2;device='npu'
        draft=torch.empty((n,old),dtype=torch.int32,device=device)
        target=torch.empty((n,old),dtype=torch.int64,device=device)
        bonus=torch.empty((n,1),dtype=torch.int32,device=device)
        narrow_draft=torch.empty((n,new),dtype=torch.int32,device=device)
        narrow_target=torch.empty((n,new),dtype=torch.int64,device=device)
        narrow_bonus=torch.empty((n,1),dtype=torch.int32,device=device)
        # The unchanged native Triton reference evaluates its offset-1 load
        # even for row0 before tl.where selects zero. Supply a guarded view,
        # not an allocation-start pointer with inaccessible unselected input.
        cu_backing=torch.zeros(n+16,dtype=torch.int32,device=device)
        ncu_backing=torch.zeros_like(cu_backing)
        cu=cu_backing[8:8+n];ncu=ncu_backing[8:8+n]
        cu.copy_(torch.arange(1,n+1,device=device,dtype=torch.int32)*old)
        ncu.copy_(torch.arange(1,n+1,device=device,dtype=torch.int32)*new)
        discarded=torch.empty(1,dtype=torch.int64,device=device)
        backup=torch.arange(30000,30000+n,dtype=torch.int32,device=device)
        owner=Obj(backup_next_token_ids=Obj(gpu=backup));batch=Obj(vocab_size=248320)
        count_args=(None,batch,discarded,seed)
        counters=torch.zeros(9,dtype=torch.int32,device=device)
        ecounters=torch.zeros_like(counters)
        full_ref=torch.empty((n,old+1),dtype=torch.int32,device=device)
        full_graph=torch.empty_like(full_ref);chain_source=torch.empty_like(full_ref)
        narrow_ref=torch.empty((n,new+1),dtype=torch.int32,device=device)
        narrow_graph=torch.empty_like(narrow_ref)
        grid,block=cal_grid_and_block_size(n)
        def update(revision):
            ds=(torch.arange(n*old,dtype=torch.int32).reshape(n,old)+1000+revision*31+seed*17)
            ts=ds.to(torch.int64)
            for row in range(n):
                mismatch=(row+revision+seed)%(old+1)
                if mismatch<old:ts[row,mismatch]+=5000
            draft.copy_(ds.to(device));target.copy_(ts.to(device))
            bonus.fill_(20000+revision+seed)
            narrow_draft.copy_(draft[:,:new]);narrow_target.copy_(target[:,:new])
            narrow_bonus.copy_(target[:,new:new+1].to(torch.int32))
            discarded.fill_(revision%n)
        def native(out,narrow=False):
            out.fill_(-1)
            rejection_greedy_sample_with_triton(out,[new if narrow else old]*n,ncu if narrow else cu,
                (narrow_draft if narrow else draft).flatten(),
                (narrow_target if narrow else target).flatten(),
                narrow_bonus if narrow else bonus,None,new if narrow else old,grid,block)
            return out
        def counts(ids):return count_tail(owner,ids,*count_args)
        def chain():
            native(chain_source)
            clipped=clip_verified_prefix(chain_source,old,new,counters)
            counted=counts(clipped)
            return clipped,counted
        update(0)
        for _ in range(3):
            native(full_ref);native(narrow_ref,True)
            chain()
        torch.npu.synchronize()
        fg=torch.npu.NPUGraph();ng=torch.npu.NPUGraph();cg=torch.npu.NPUGraph()
        with torch.npu.graph(fg):native(full_graph)
        with torch.npu.graph(ng):native(narrow_graph,True);held_ncounts=counts(narrow_graph)
        with torch.npu.graph(cg):held,held_counts=chain()
        sources=(draft,target,bonus,narrow_draft,narrow_target,narrow_bonus,cu_backing,ncu_backing,cu,ncu,discarded,backup)
        for revision in range(9):
            update(revision);copies=[t.clone() for t in sources]
            native(full_ref);native(narrow_ref,True);torch.npu.synchronize()
            fg.replay();ng.replay();torch.npu.synchronize()
            assert torch.equal(full_ref,full_graph) and torch.equal(narrow_ref,narrow_graph),'native self graph'
            original=full_ref.clone();ecounters.zero_()
            eager=clip_verified_prefix(full_ref,old,new,ecounters)
            enext,ecount=counts(eager);nnext,ncount=counts(narrow_ref)
            counters.zero_();cg.replay();torch.npu.synchronize()
            expected=torch.full_like(full_ref,-1);expected[:,:new+1]=narrow_ref
            assert torch.equal(eager,expected) and torch.equal(held,expected),'independent native narrow reference'
            assert torch.equal(full_ref,original) and torch.equal(chain_source,full_ref),'native wide output readonly'
            assert eager.data_ptr()!=full_ref.data_ptr() and held.data_ptr()!=chain_source.data_ptr()
            assert torch.equal(enext,nnext) and torch.equal(ecount,ncount),'native count and next token'
            assert torch.equal(held_counts[0],held_ncounts[0]) and torch.equal(held_counts[1],held_ncounts[1])
            wanted=torch.zeros_like(counters);wanted[old]=1
            assert torch.equal(counters,wanted) and torch.equal(ecounters,wanted),'complete folded witness'
            assert all(torch.equal(x,y) for x,y in zip(sources,copies)),'all original inputs readonly'
        # No diagnostic synchronization between operations/calls. Prove the
        # intended async eager producer->clip->original-count order too.
        counters.zero_()
        for _ in range(32):stress_output,stress_counts=chain()
        torch.npu.synchronize()
        assert torch.equal(stress_output,expected)
        assert torch.equal(stress_counts[0],nnext) and torch.equal(stress_counts[1],ncount)
        assert torch.equal(chain_source,full_ref)
        wanted.zero_();wanted[old]=32;assert torch.equal(counters,wanted)
        assert all(torch.equal(x,y) for x,y in zip(sources,copies))
        records.append(dict(requests=n,old=old,new=new,discarded_requests=seed,
            nine_revisions_including_all_rejection_positions_and_all_accept=True,
            independent_native_wide_and_narrow_eager_self_graph=True,
            all_output_ID_bits_and_native_count_next_ID_bits_exact=True,
            original_wide_output_and_all_inputs_readonly=True,
            complete_in_kernel_folded_witness_exact=True,
            asynchronous_eager_chain_32calls_without_intermediate_sync_verified=True,
            native_reference_CU_guarded_backing_unselected_offset_minus1_read_explicit=True,
            no_attention_KV_SSM_or_existing_count_tensor_access=True))
        print(json.dumps(dict(device=a.device,**records[-1])),flush=True)
        del fg,ng,cg,held,held_counts,held_ncounts,copies,eager,expected,original
        torch.npu.empty_cache()
    paths=[*ROOT.glob('*.py'),SOURCE,Path(native_sample_module.__file__),
        Path(native_sample_module.__file__).parent/'triton_utils.py',
        Path(native_sample_module.__file__).parents[2]/'device/device_op.py']
    write=dict(completed=True,device=a.device,records=records,component_only=True,
        native_GPU_count_tail_excludes_original_CPU_backup_setup_explicitly=True,
        NOT_model_live_bridge_or_quality_or_speed_or_goal_admission=True,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with a.output.open('x') as f:json.dump(write,f,indent=2)

if __name__=='__main__':main()
