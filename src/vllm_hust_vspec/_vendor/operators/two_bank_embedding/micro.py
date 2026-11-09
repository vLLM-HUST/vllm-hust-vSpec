"""Both allocated NPUs, real checkpoint rows, full-vocab bits and graph/timing controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
os.umask(0o077)
ROOT=Path(__file__).resolve().parent
MODEL=Path(os.environ['VSPEC_MODEL_DIR'])
import torch
import torch_npu
from safetensors import safe_open
from kernel import embedding
parser=argparse.ArgumentParser();parser.add_argument('--logical',type=int,choices=(0,1),required=True)
args=parser.parse_args()
from vllm_hust_vspec.backends.experimental_paths import current_npu_allocation
physical_card=current_npu_allocation()[args.logical]
assert not os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
assert torch.npu.device_count()==2
torch.npu.set_device(args.logical);device=f'npu:{args.logical}'
index=MODEL/'model.safetensors.index.json';mapping=json.loads(index.read_text())['weight_map']
key='model.language_model.embed_tokens.weight'
checkpoint=MODEL/mapping[key]
with safe_open(checkpoint,framework='pt',device='cpu') as source:
    cpu=source.get_tensor(key)
assert cpu.shape==(248320,2048) and cpu.dtype==torch.bfloat16 and cpu.is_contiguous()
tensor_sha256=hashlib.sha256(cpu.view(torch.uint16).numpy().tobytes()).hexdigest()
half=124160
reference=cpu.to(device)
local=cpu[args.logical*half:(args.logical+1)*half].to(device)
remote=cpu[(1-args.logical)*half:(2-args.logical)*half].to(device)
assert all(torch_npu.get_npu_format(t) in (0,2) for t in (reference,local,remote))
assert local.untyped_storage().data_ptr()!=remote.untyped_storage().data_ptr()
assert reference.untyped_storage().data_ptr() not in {local.untyped_storage().data_ptr(),remote.untyped_storage().data_ptr()}
checks=[]
counter=torch.zeros(8193,device=device,dtype=torch.int32)
for block in (512,1024,2048):
    for start in range(0,248320,4096):
        ids=torch.arange(start,min(start+4096,248320),dtype=torch.int64,device=device)
        actual=embedding(ids,local,remote,args.logical,block,counter=counter)
        expected=torch.nn.functional.embedding(ids,reference)+0.
        assert torch.equal(actual.view(torch.int16),expected.view(torch.int16)),(block,start)
    checks.append(dict(block=block,all248320_vocabulary_rows_bitwise_exact=True))
    print(json.dumps(dict(physical_card=physical_card,block=block,all_vocab_passed=True)),flush=True)
# Signed zeros are exact in both halves; reference remains a separate immutable table.
# Separate synthetic small banks do not alter real checkpoint tensors.
synthetic=torch.zeros((32,2048),device=device,dtype=torch.bfloat16)
synthetic[0].fill_(-0.);synthetic[16].fill_(-0.)
ids=torch.tensor([0,1,15,16,17,31],device=device)
left,right=synthetic[:16].contiguous(),synthetic[16:].contiguous()
local_s,remote_s=(left,right) if args.logical==0 else (right,left)
assert torch.equal(embedding(ids,local_s,remote_s,args.logical).view(torch.int16),
                   (torch.nn.functional.embedding(ids,synthetic)+0.).view(torch.int16))
retained=[];records=[]
for rows in (1,2,3,7,14,16,21,35,70,98,208,8192):
    ids_cpu=((torch.arange(rows,dtype=torch.int64)*104729+124159)%248320)
    ids=ids_cpu.to(device)
    # Do not compare aliased output buffers or discard earlier graphs/banks during this probe.
    graphs=[]
    for kind,block in [('native_full',None),('split512',512),('split1024',1024),('split2048',2048)]:
        fn=(lambda:torch.nn.functional.embedding(ids,reference)+0.) if block is None else (lambda b=block:embedding(ids,local,remote,args.logical,b,counter=counter))
        for _ in range(3):out=fn()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):output=fn()
        graphs.append((kind,graph,output));retained.append((ids,graph,output))
    expected=torch.nn.functional.embedding(ids,reference)+0.
    for kind,graph,output in graphs:
        for _ in range(8):
            graph.replay();torch.npu.synchronize()
            assert torch.equal(output.view(torch.int16),expected.view(torch.int16)),(rows,kind)
    timings={kind:[] for kind,_,_ in graphs}
    iterations=64 if rows<=208 else 2
    for repeat in range(8):
        order=graphs[repeat%len(graphs):]+graphs[:repeat%len(graphs)]
        for kind,graph,output in order:
            start,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):graph.replay()
            end.record();end.synchronize()
            timings[kind].append(start.elapsed_time(end)*1000/iterations)
    import statistics
    row=dict(rows=rows,timing_graph_replays_per_sample=iterations,us={kind:statistics.median(values) for kind,values in timings.items()},
             eager_and8_graph_revisions_all_consumed_rows_bitwise_exact=True)
    records.append(row);print(json.dumps(row),flush=True)
paths=[ROOT/'kernel.py',Path(__file__),index]
actual_counts=counter.cpu().tolist()
assert all(actual_counts[n]>0 for n in (1,2,3,7,14,16,21,35,70,98,208,8192))
proof=dict(passed=True,diagnostic_only=True,physical_npus=[physical_card],all_vocab_checks=checks,
           actual_mutating_counter_per_rows={str(n):v for n,v in enumerate(actual_counts) if v},
           numeric_and_graph_timing_checks_include_same_counter_required_in_model=True,
           checkpoint_key=key,checkpoint_file=str(checkpoint),embedding_tensor_sha256=tensor_sha256,
           reference='existing_full_table_F_embedding_plus_positive_zero_NOT_original_TP_collective',
           original_native_TP_model_startup_and_full_model_shadows_still_required=True,
           synthetic_signed_zero_both_halves_exact=True,records=records,
           logical_native_local_partition_bytes=half*2048*2,
           proposed_ON_extra_persistent_bytes_remote_half_only=half*2048*2,
           saved_vs_full_replica_plus_native_partition_bytes=half*2048*2,
           no_checkpoint_or_target_weights_modified=True,no_end_to_end_speed_or_quality_claim=True,
           sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
with (ROOT/f'split-embedding-card{args.logical+1}-r1.json').open('x') as f:json.dump(proof,f,indent=2)
