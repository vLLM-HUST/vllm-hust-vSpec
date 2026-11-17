"""Unselected gamma6 RGDR: independent native eager/graph and full state proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
ORIGINAL=Path(os.environ['VSPEC_GAMMA6_RGDR_REFERENCE_DIR'])
VARIANTS=dict(current128=('vspec_rgdr_tail6',128,1),packed128=('vspec_rgdr_pack6',128,1))

def length_patterns(n):
    assert 1<=n<=16
    uniform=[7]*n
    uneven=([6,8]*(n//2))+([7] if n%2 else [])
    padded=([0,12]+[7]*(n-3)+[9]) if n>=3 else uneven
    return [uniform,uneven,padded]

def main():
    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);torch.set_num_threads(1)
    torch.ops.load_library(str(ROOT/'build-r1/lib/libvspec_rgdr_pack6_adapter.so'))
    torch.ops.load_library(str(ORIGINAL/'build-r1/lib/libvspec_rgdr_tail6_adapter.so'))
    native=torch.ops._C_ascend.npu_recurrent_gated_delta_rule
    records=[]
    for n,bank in ((n,b) for n in range(1,17) for b in (192,647)):
        tokens=n*7
        for seed in range(6):
            torch.manual_seed(11048000+seed+rank*100)
            q=torch.empty((tokens,8,128),dtype=torch.bfloat16,device='npu');k=torch.empty_like(q)
            v=torch.empty((tokens,16,128),dtype=torch.bfloat16,device='npu')
            beta=torch.empty((tokens,16),dtype=torch.bfloat16,device='npu')
            g=torch.empty((tokens,16),dtype=torch.float32,device='npu')
            initial=torch.randn((bank,16,128,128),dtype=torch.float32,device='npu')/8
            initial.view(torch.int32)[-1].fill_(2143354881)
            initial.view(torch.int32)[-2,0,0,0]=-2147483648
            max_offset=bank-tokens-2
            offset=tuple(min(x,max_offset) for x in ((0,7,29,60) if bank==192 else (0,128,256,480)))[seed%4]
            indices=(torch.randperm(tokens,device='npu')+offset).to(torch.int32)
            routes=('uniform','uneven','padded_zero')
            patterns=length_patterns(n)
            route=routes[seed//2] if n>=3 else ('uniform' if seed//2==0 else 'uneven_no_zero_possible')
            lens=patterns[seed//2]
            assert len(lens)==n and sum(lens)==tokens and max(lens)<=16
            lengths=torch.tensor([0]+lens,dtype=torch.int32,device='npu')
            accepted=torch.empty(n,dtype=torch.int32,device='npu')
            arms=('native',*VARIANTS)
            states={arm:initial.clone() for arm in arms}
            counts={arm:torch.zeros(17,dtype=torch.int32,device='npu') for arm in arms}
            def reset(arm):states[arm].copy_(initial);counts[arm].zero_()
            def inputs(revision):
                lens=patterns[(seed//2+revision)%3]
                lengths.copy_(torch.tensor([0]+lens,dtype=torch.int32,device='npu'))
                q.copy_(torch.nn.functional.normalize(torch.randn(q.shape,device='npu'),dim=-1))
                k.copy_(torch.nn.functional.normalize(torch.randn(k.shape,device='npu'),dim=-1))
                v.normal_(0,(.125,1,2,4,.01,.5,8,0)[revision])
                beta.uniform_(0,1);g.uniform_(-.5,0)
                accepted.copy_(torch.tensor([0 if n==0 else 1+(i+seed+revision)%n for i,n in enumerate(lens)],dtype=torch.int32,device='npu'))
            def invoke(arm):
                if arm=='native':return native(query=q,key=k,value=v,state=states[arm],beta=beta,g=g,
                    scale=128**-.5,actual_seq_lengths=lengths,ssm_state_indices=indices,num_accepted_tokens=accepted)
                namespace,step,buffers=VARIANTS[arm]
                return getattr(torch.ops,namespace).project(q,k,v,beta,states[arm],lengths,indices,g,accepted,counts[arm],step,buffers)
            def check(arm,output,expected,state,phase):
                torch.npu.synchronize()
                assert torch.isfinite(output).all() and torch.isfinite(expected).all(),(bank,seed,arm,phase,'finite')
                assert torch.equal(output.view(torch.int16),expected.view(torch.int16)),(bank,seed,arm,phase,'all output bits')
                assert torch.equal(states[arm].view(torch.int32),state.view(torch.int32)),(bank,seed,arm,phase,'FULL physical FP32 state')
                wanted=torch.zeros_like(counts[arm])
                if arm!='native':wanted[n]=1
                assert torch.equal(counts[arm],wanted),(bank,seed,arm,phase,'all counter cells')
            inputs(0);graphs={}
            for arm in arms:
                for _ in range(3):reset(arm);invoke(arm)
                torch.npu.synchronize();reset(arm);graph=torch.npu.NPUGraph()
                with torch.npu.graph(graph):held=invoke(arm)
                graphs[arm]=(graph,held)
            for revision in range(8):
                inputs(revision)
                sources=(q,k,v,g,beta,lengths,indices,accepted)
                snapshots=[t.clone() for t in sources]
                reset('native');expected=invoke('native');torch.npu.synchronize()
                expected_state=states['native'].clone()
                reset('native');graphs['native'][0].replay()
                check('native',graphs['native'][1],expected,expected_state,'independent_native_graph')
                for arm in VARIANTS:
                    reset(arm);output=invoke(arm);check(arm,output,expected,expected_state,'eager')
                    reset(arm);graphs[arm][0].replay();check(arm,graphs[arm][1],expected,expected_state,'graph')
                for x,y in zip(sources,snapshots):
                    dtype=torch.int16 if x.dtype==torch.bfloat16 else torch.int32
                    assert torch.equal(x.view(dtype),y.view(dtype)),(bank,seed,revision,'complete readonly input bits')
            # Continuous state progression, NOT reset-to-reference between
            # rounds: catch any stale-cache read after a differently hinted
            # write, including graph replay and dynamic native padding.
            for arm in arms:reset(arm)
            for revision in range(12):
                inputs(revision%8);snapshots=[t.clone() for t in sources]
                counts['native'].zero_();graphs['native'][0].replay();torch.npu.synchronize()
                expected=graphs['native'][1].clone();expected_state=states['native'].clone()
                for arm in VARIANTS:
                    counts[arm].zero_();graphs[arm][0].replay()
                    check(arm,graphs[arm][1],expected,expected_state,'continuous_state_NOT_reset')
                for x,y in zip(sources,snapshots):
                    dtype=torch.int16 if x.dtype==torch.bfloat16 else torch.int32
                    assert torch.equal(x.view(dtype),y.view(dtype)),(bank,seed,revision,'continuous readonly input bits')
            # Time nonzero representative values, not the final zero-v edge.
            inputs(1);timed_sources=[t.clone() for t in sources]
            samples={arm:[] for arm in arms}
            for repeat in range(2*len(arms)):
                for arm in arms[repeat%len(arms):]+arms[:repeat%len(arms)]:
                    reset(arm);begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    begin.record()
                    for _ in range(30):graphs[arm][0].replay()
                    end.record();end.synchronize();samples[arm].append(begin.elapsed_time(end)*1000/30)
            for x,y in zip(sources,timed_sources):
                dtype=torch.int16 if x.dtype==torch.bfloat16 else torch.int32
                assert torch.equal(x.view(dtype),y.view(dtype)),(bank,seed,'timed inputs readonly')
            record=dict(bank=bank,seed=seed,width=7,requests=n,route=route,
                timed_route_pattern_index=(seed//2+1)%3,
                zero_length_padding_possible=n>=3,
                timing_uses_nonzero_vscale1_inputs=True,
                independent_native_eager_graph_reference=True,eight_eager_graph_revisions_bits_exact=True,
                captured_graph_dynamic_lengths_and_accepted_covered=True,
                twelve_continuous_state_replays_WITHOUT_state_reset_full_bits_exact=True,
                ONLY_delta_replication_layout_changed_FMA_operand_order_and_K128_reductions_unchanged=True,
                original_single_state_output_queue_all_buffers_and_events_UNCHANGED=True,
                SAME_numerical_arithmetic_FP32_state_and_BF16_inputs_no_persisted_KV=True,
                full_FP32_state_including_untouched_NaN_payload_and_signed_zero_exact=True,
                complete_inputs_bits_readonly=True,entire_folded_counter_array_exact=True,
                median_us={arm:statistics.median(value) for arm,value in samples.items()})
            records.append(record);print(json.dumps(dict(rank=rank,**record)),flush=True)
            del graphs,graph,held,expected,expected_state,initial,snapshots,timed_sources,output
            states.clear();counts.clear();torch.npu.empty_cache()
    paths=[*ROOT.glob('*.py'),*ROOT.glob('*.cpp'),*ROOT.glob('*.h'),ROOT/'CMakeLists.txt',
        ROOT/'build-r1/lib/libvspec_rgdr_pack6_adapter.so',ROOT/'build-r1/lib/libvspec_rgdr_pack6_kernel.so',
        ORIGINAL/'build-r1/lib/libvspec_rgdr_tail6_adapter.so',ORIGINAL/'build-r1/lib/libvspec_rgdr_tail6_kernel.so',
        ORIGINAL/'runs/gamma6-tail-c46-r1.primitive-audit.json']
    with a.output.with_name(a.output.stem+f'-rank{rank}.json').open('x') as f:
        json.dump(dict(completed=True,rank=rank,records=records,component_only=True,
            gamma4_and8_not_dispatched=True,no_precision_change=True,
            sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),f,indent=2)
if __name__=='__main__':main()
