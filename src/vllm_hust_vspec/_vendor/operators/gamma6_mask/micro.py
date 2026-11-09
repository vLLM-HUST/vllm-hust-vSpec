"""Native eager/self-graph versus old separate-count and new folded witness."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parent
OLD=Path(os.environ['VSPEC_ASCEND_SOURCE_ROOT'])/'vllm_ascend/ops/triton/q35_expert_mask.py'
ARMS=('native','old_separate_count','fused_folded_count')

def signed(word,bits):return word-(1<<bits) if word>=(1<<(bits-1)) else word

def main():
    import torch
    import torch_npu
    from kernel import fused
    spec=importlib.util.spec_from_file_location('unchanged_original_mask',OLD)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank);torch.set_num_threads(1)
    records=[]
    for n in range(1,17):
        rows=n*7;size=rows*8
        for id_dtype in (torch.int32,torch.int64):
            for weight_dtype in (torch.bfloat16,torch.float32):
                torch.manual_seed(11046000+rank*100+n)
                id_base=torch.full((size+32,),-123456,dtype=id_dtype,device='npu')
                weight_base=torch.full((size+32,),.03125,dtype=weight_dtype,device='npu')
                map_base=torch.full((288,),-123456,dtype=torch.int32,device='npu')
                ids=id_base[16:-16].view(rows,8);weights=weight_base[16:-16].view(rows,8)
                expert_map=map_base[16:-16]
                counts={arm:torch.zeros(17,dtype=torch.int32,device='npu') for arm in ARMS}
                special_bf16=(0x0000,0x8000,0x3f80,0xbf80,0x0001,0x8001,0x007f,0x807f,
                    0x0080,0x8080,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffc1,0x7f81,0xff81)
                special_fp32=(0x00000000,0x80000000,0x3f800000,0xbf800000,0x00000001,0x80000001,
                    0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff,
                    0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7f800001,0xff800001)
                def inputs(phase,special=True):
                    values=(torch.arange(size,device='npu')*13+phase*29+rank*7)%256
                    ids.copy_(values.reshape_as(ids).to(id_dtype))
                    mapping=torch.arange(256,device='npu',dtype=torch.int32)%128
                    if phase%4==0:mapping.fill_(-1)
                    elif phase%4!=1:mapping[(torch.arange(256,device='npu')+phase)%3==0]=-1
                    expert_map.copy_(mapping)
                    weights.copy_(torch.randn(weights.shape,device='npu').to(weight_dtype))
                    if special:
                        patterns=special_bf16 if weight_dtype==torch.bfloat16 else special_fp32
                        width=16 if weight_dtype==torch.bfloat16 else 32
                        raw=torch.tensor([signed(patterns[(i+phase)%len(patterns)],width) for i in range(size)],
                            dtype=torch.int16 if width==16 else torch.int32,device='npu')
                        weights.view(raw.dtype).copy_(raw.view_as(weights))
                def invoke(arm):
                    if arm=='native':return weights*(expert_map[ids]!=-1)
                    if arm=='old_separate_count':
                        result=old.fused_mask(ids,weights,expert_map);counts[arm][n].add_(1);return result
                    return fused(ids,weights,expert_map,counts[arm])
                def check(arm,actual,expected):
                    torch.npu.synchronize()
                    dtype=torch.int16 if weight_dtype==torch.bfloat16 else torch.int32
                    assert torch.equal(actual.view(dtype),expected.view(dtype)),(rank,n,str(id_dtype),str(weight_dtype),arm,'ALL raw output bits including NaN/signed0')
                    wanted=torch.zeros_like(counts[arm])
                    if arm!='native':wanted[n]=1
                    assert torch.equal(counts[arm],wanted),(rank,n,arm,'all17 folded counter cells')
                inputs(0,False);graphs={}
                for arm in ARMS:
                    for _ in range(3):invoke(arm)
                    torch.npu.synchronize();counts[arm].zero_();graph=torch.npu.NPUGraph()
                    with torch.npu.graph(graph):held=invoke(arm)
                    graphs[arm]=(graph,held)
                for phase in range(8):
                    inputs(phase)
                    bases=(id_base,weight_base,map_base);saved=[x.clone() for x in bases]
                    counts['native'].zero_();expected=invoke('native');torch.npu.synchronize()
                    counts['native'].zero_();graphs['native'][0].replay();check('native',graphs['native'][1],expected)
                    for arm in ARMS[1:]:
                        counts[arm].zero_();actual=invoke(arm);check(arm,actual,expected)
                        counts[arm].zero_();graphs[arm][0].replay();check(arm,graphs[arm][1],expected)
                    for source,prior in zip(bases,saved):
                        dtype=torch.int16 if source.dtype==torch.bfloat16 else torch.int32 if source.dtype==torch.float32 else source.dtype
                        assert torch.equal(source.view(dtype),prior.view(dtype)),(rank,n,'full input storage including canaries readonly')
                inputs(3,False);saved=[x.clone() for x in bases]
                samples={arm:[] for arm in ARMS}
                for repeat in range(7):
                    order=ARMS[repeat%len(ARMS):]+ARMS[:repeat%len(ARMS)]
                    for arm in order:
                        begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
                        begin.record()
                        for _ in range(100):graphs[arm][0].replay()
                        end.record();end.synchronize();samples[arm].append(begin.elapsed_time(end)*1000/100)
                for source,prior in zip(bases,saved):
                    dtype=torch.int16 if source.dtype==torch.bfloat16 else torch.int32 if source.dtype==torch.float32 else source.dtype
                    assert torch.equal(source.view(dtype),prior.view(dtype))
                record=dict(requests=n,tokens=rows,id_dtype=str(id_dtype),weight_dtype=str(weight_dtype),
                    independent_native_eager_self_graph_bits_exact=True,eight_dynamic_input_eager_graph_revisions_bits_exact=True,
                    all_raw_output_bits_NaN_payload_signed_zero_infinity_subnormal_covered=True,
                    complete_input_storage_including_canaries_readonly=True,all17_counter_cells_exact=True,
                    only_post_native_top8_lookup_predicate_multiply=True,router_scores_top8_and_weight_values_unchanged=True,
                    native_negative_and_OOB_ID_semantics_NOT_claimed=True,
                    timing_finite_random_weights_not_special_values=True,
                    samples_us=samples,median_us={arm:statistics.median(value) for arm,value in samples.items()})
                records.append(record);print(json.dumps(dict(rank=rank,**record)),flush=True)
                del graphs,graph,held,actual,expected,saved
                counts.clear();torch.npu.empty_cache()
    # Hash only immutable test inputs. Other ranks may still be writing results.
    paths=[ROOT/'micro.py',ROOT/'kernel.py',OLD]
    output=args.output.with_name(args.output.stem+f'-rank{rank}.json')
    with output.open('x') as stream:json.dump(dict(passed=True,rank=rank,records=records,
        component_only_NOT_model_speed_quality_or_goal_admission=True,
        sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}),stream,indent=2)

if __name__=='__main__':main()
