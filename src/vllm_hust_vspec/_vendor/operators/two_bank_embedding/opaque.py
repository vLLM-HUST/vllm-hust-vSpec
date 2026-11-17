"""Opaque mutable-counter ABI for the bit-proved two-bank proposal lookup."""
import importlib.util
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent
NAME='_vspec_two_bank_primitive_kernel'
if NAME not in sys.modules:
    spec=importlib.util.spec_from_file_location(NAME,ROOT/'kernel.py')
    module=importlib.util.module_from_spec(spec);sys.modules[NAME]=module;spec.loader.exec_module(module)
kernel=sys.modules[NAME]
_DEF=torch.library.Library('vspec_split_embedding','DEF')
_DEF.define('lookup(Tensor ids, Tensor local, Tensor remote, Tensor(a!) counter, int rank, int block) -> Tensor')
_IMPL=torch.library.Library('vspec_split_embedding','IMPL','PrivateUse1')

def _lookup(ids,local,remote,counter,rank,block):
    return kernel.embedding(ids,local,remote,rank,block,counter=counter)
_IMPL.impl('lookup',_lookup)

@torch.library.register_fake('vspec_split_embedding::lookup')
def _fake(ids,local,remote,counter,rank,block):
    return torch.empty((ids.numel(),local.shape[1]),device=ids.device,dtype=local.dtype)
