"""Unselected component experiment: native gamma2/8 convolution, full-state proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from itertools import product

ROOT=Path(__file__).resolve().parent
BATCHES=tuple(range(1,17))
WIDTHS=(3,9)
ROUTES=('uniform','uneven','padded_zero')
TYPES=('i32','i64','mixed')
LAYOUTS=('contiguous','qkvz_split_6144')
TILES=(1024,2048)

def main():
    import torch
    import torch_npu
    import vllm_ascend.vllm_ascend_C
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);torch.set_num_threads(1)
    torch.manual_seed(11042026+rank)
    torch.ops.load_library(str(ROOT/'build-r1/lib/libvspec_even_edge_adapter.so'))
    records=[]
    for n,width,route,types,layout in product(BATCHES,WIDTHS,ROUTES,TYPES,LAYOUTS):
        integer=torch.int64 if types in ('i64','mixed') else torch.int32
        index_integer=torch.int64 if types=='i64' else torch.int32
        tokens=n*width;columns=4096 if layout=='contiguous' else 6144
        base=torch.empty((tokens,columns),dtype=torch.bfloat16,device='npu');value=base[:,:4096]
        if columns>4096:
            base[:,4096:].view(torch.int16).fill_(32705);base[:,-1].view(torch.int16).fill_(-32768)
        weight=(torch.randn((4,4096),device='npu')/16).to(torch.bfloat16)
        initial=(torch.randn((192,11,4096),device='npu')/16).to(torch.bfloat16)
        initial.view(torch.int16)[191].fill_(32705);initial.view(torch.int16)[190,-1,-1]=-32768
        lens=[width]*n
        if route=='uneven':lens=[width-int(i%2==0) for i in range(n)]
        elif route=='padded_zero':lens[0]=0
        prefix=[0]
        for length in lens:prefix.append(prefix[-1]+length)
        queries=torch.tensor(prefix,dtype=integer,device='npu')
        indices=torch.arange(n*9,dtype=index_integer,device='npu').reshape(n,9).flip(0).contiguous()
        accepted=torch.empty(n,dtype=integer,device='npu')
        arms=('native',*map(str,TILES))
        states={arm:initial.clone() for arm in arms}
        outputs={arm:torch.empty_like(value) for arm in arms}
        counters={arm:torch.zeros(34,dtype=torch.int32,device='npu') for arm in arms}
        def reset(arm):
            states[arm].copy_(initial);outputs[arm].view(torch.int16).fill_(32705);counters[arm].zero_()
        def invoke(arm):
            if arm=='native':
                torch.ops._C_ascend.npu_causal_conv1d_custom(outputs[arm],value,weight,conv_state=states[arm],
                    bias_opt=None,query_start_loc_opt=queries,cache_indices_opt=indices,initial_state_mode_opt=None,
                    num_accepted_tokens_opt=accepted,activation_mode=1,pad_slot_id=-1,run_mode=1)
            else:torch.ops.vspec_even_edge_conv.project(outputs[arm],value,weight,states[arm],queries,indices,accepted,counters[arm],int(arm))
        def check(arm,expected,state,phase):
            assert torch.equal(outputs[arm].view(torch.int16),expected.view(torch.int16)),(n,width,route,types,layout,arm,phase,'output')
            assert torch.equal(states[arm].view(torch.int16),state.view(torch.int16)),(n,width,route,types,layout,arm,phase,'FULL PHYSICAL STATE')
            wanted=torch.zeros_like(counters[arm])
            if arm!='native':wanted[n+(17 if width==9 else 0)]=1
            assert torch.equal(counters[arm],wanted),(n,width,arm,phase,'all counter cells')
        value.normal_(0,.0625);accepted.fill_(1);graphs={}
        for arm in arms:
            for _ in range(3):reset(arm);invoke(arm)
            torch.npu.synchronize();reset(arm)
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph):invoke(arm)
            graphs[arm]=graph
        for revision in range(8):
            value.normal_(0,(.0625,.5,1,4,.01,8,.125,2)[revision])
            accepted.copy_(torch.tensor([0 if length==0 else 1+(i+revision)%length for i,length in enumerate(lens)],dtype=integer,device='npu'))
            sources=[x.clone() for x in (base,weight,queries,indices,accepted)]
            reset('native');invoke('native');torch.npu.synchronize()
            expected,state=outputs['native'].clone(),states['native'].clone()
            reset('native');graphs['native'].replay();torch.npu.synchronize();check('native',expected,state,'independent_native_graph')
            for arm in arms[1:]:
                reset(arm);invoke(arm);torch.npu.synchronize();check(arm,expected,state,'eager')
                reset(arm);graphs[arm].replay();torch.npu.synchronize();check(arm,expected,state,'graph')
            assert all(torch.equal(x.view(torch.int16) if x.dtype==torch.bfloat16 else x,
                                   y.view(torch.int16) if y.dtype==torch.bfloat16 else y)
                       for x,y in zip((base,weight,queries,indices,accepted),sources))
        medians={}
        if n in (1,8,14,16) and route=='uniform' and types=='i32':
            samples={arm:[] for arm in arms}
            for repeat in range(5):
                order=arms[repeat%len(arms):]+arms[:repeat%len(arms)]
                for arm in order:
                    reset(arm);start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                    start.record()
                    for _ in range(40):graphs[arm].replay()
                    end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)*1000/40)
            medians={arm:statistics.median(x) for arm,x in samples.items()}
        record=dict(n=n,width=width,route=route,types=types,layout=layout,state_length=11,index_stride=9,
            eight_eager_and_graph_revisions_bits_exact=True,FULL_native_physical_state_bits_exact=True,
            independent_native_graph_passed=True,whole_source_base_weights_metadata_readonly=True,
            all_folded_replay_counter_cells_exact=True,median_us=medians)
        records.append(record);print(json.dumps(dict(rank=rank,**record)),flush=True)
        del graphs,graph,expected,state,initial,sources
        states.clear();outputs.clear();torch.npu.empty_cache()
    paths=[ROOT/x for x in ('micro.py','adapter.cpp','kernel.cpp','CMakeLists.txt','build-r1/lib/libvspec_even_edge_adapter.so','build-r1/lib/libvspec_even_edge_kernel.so')]
    paths.extend(sorted((ROOT/'native-kernel').rglob('*.h')))
    output=a.output.with_name(a.output.stem+f'-rank{rank}.json')
    with output.open('x') as f:json.dump(dict(completed=True,rank=rank,records=records,
        standalone_component_only_NOT_model_or_task_quality_or_end_to_end_speed_admission=True,
        gamma4_and6_not_dispatched_by_this_operator=True,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),f,indent=2)

if __name__=='__main__':main()
