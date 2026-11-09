"""Real dispatcher, fake dynamic fullgraph and actual NPU capture mutation checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
os.umask(0o077)
ROOT=Path(__file__).resolve().parent
import torch
import torch_npu
import opaque
parser=argparse.ArgumentParser();parser.add_argument('--logical',type=int,choices=(0,1),required=True)
args=parser.parse_args()
from vllm_hust_vspec.backends.experimental_paths import current_npu_allocation
physical_card=current_npu_allocation()[args.logical]
assert not os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
assert torch.npu.device_count()==2
proof_path=ROOT/'split-embedding-r1-primitive-audit.json'
parent=json.loads(proof_path.read_text())
assert parent['passed'] and all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in parent['sha256'].items())
schema=str(torch.ops.vspec_split_embedding.lookup.default._schema)
assert 'Tensor(a!) counter' in schema and '-> Tensor' in schema
# Before any NPU context: fullgraph tracing through the real fake implementation.
from torch._subclasses.fake_tensor import FakeTensorMode
compiled_graphs=[]
def backend(gm,example_inputs):
    assert 'vspec_split_embedding.lookup' in str(gm.graph)
    compiled_graphs.append(str(gm.graph));return gm.forward
f=torch.compile(lambda ids,l,r,c:torch.ops.vspec_split_embedding.lookup(ids,l,r,c,0,2048),
                backend=backend,dynamic=True,fullgraph=True)
with FakeTensorMode():
    l=torch.empty((124160,2048),device='meta',dtype=torch.bfloat16)
    r=torch.empty_like(l);c=torch.empty(8193,device='meta',dtype=torch.int32)
    for n in (8192,70,98):
        ids=torch.empty(n,device='meta',dtype=torch.int64)
        out=f(ids,l,r,c)
        assert out.shape==(n,2048) and out.dtype==torch.bfloat16 and out is not c
assert len(compiled_graphs)==1,len(compiled_graphs)
torch._dynamo.reset()
torch.npu.set_device(args.logical);device=f'npu:{args.logical}'
full=((torch.arange(64*2048,device=device).reshape(64,2048)%127-63)/128).to(torch.bfloat16)
full[0].fill_(-0.);full[32].fill_(-0.)
lo=full[:32].clone();hi=full[32:].clone()
local,remote=(lo,hi) if args.logical==0 else (hi,lo)
counter=torch.zeros(8193,device=device,dtype=torch.int32)
real_graphs=[]
def real_backend(gm,example_inputs):
    assert 'vspec_split_embedding.lookup' in str(gm.graph)
    real_graphs.append(str(gm.graph));return gm.forward
compiled=torch.compile(lambda ids:torch.ops.vspec_split_embedding.lookup(ids,local,remote,counter,args.logical,2048),
                       backend=real_backend,dynamic=True,fullgraph=True)
retained=[];records=[]
for rows in (14,70,98):
    ids=(torch.arange(rows,device=device,dtype=torch.int64)*17)%64
    expected=torch.nn.functional.embedding(ids,full)+0.
    actual=compiled(ids)
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph):output=compiled(ids)
    graph.replay();torch.npu.synchronize()
    assert torch.equal(output.view(torch.int16),expected.view(torch.int16))
    before=int(counter[rows].cpu())
    for _ in range(5):graph.replay()
    torch.npu.synchronize()
    after=int(counter[rows].cpu())
    assert after-before==5,(rows,before,after)
    assert torch.equal(output.view(torch.int16),expected.view(torch.int16))
    records.append(dict(rows=rows,compiled_NPU_output_bits_exact=True,captured_graph_output_bits_exact=True,
                        mutable_counter_delta_for5_replays=after-before,signed_zero_inputs_included=True))
    retained.append((ids,graph,output))
assert len(real_graphs)==1,len(real_graphs)
ps=[Path(__file__),ROOT/'opaque.py',ROOT/'kernel.py',proof_path]
with (ROOT/f'opaque-card{args.logical+1}-r1.json').open('x') as file:
    json.dump(dict(passed=True,diagnostic_only=True,physical_npus=[physical_card],
        actual_mutating_ABI=schema,real_fake_Meta_and_NPU_fullgraph_each_one_graph_dynamic70_98=True,
        records=records,model_integration_and_native_TP_startup_and_quality_still_required=True,
        sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}),file,indent=2)
print(json.dumps(dict(passed=True,physical_card=physical_card,records=records)),flush=True)
