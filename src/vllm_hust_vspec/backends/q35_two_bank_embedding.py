"""Default-off proposal-only remote-half replica; shared target weights untouched."""
from functools import wraps
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType

from ..config import q35_two_bank_embedding_enabled,q35_replicated_inputs
from .experimental_paths import artifact_dir, require_current_npu_proof

PRIMITIVE=artifact_dir('VSPEC_TWO_BANK_EMBEDDING_DIR')
HALF=124160
WIDTH=2048
MAX_ROWS=8192
BLOCK=2048


def verify_primitive():
    paths=[PRIMITIVE/'split-embedding-r1-primitive-audit.json',PRIMITIVE/'opaque-r1-audit.json']
    for path in paths:
        d=json.loads(path.read_text())
        require_current_npu_proof(d)
        assert d['passed']
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in d['sha256'].items()),path
        for row in d.get('records',[]):
            if isinstance(row,dict) and 'sha256' in row:
                assert row['passed']
                assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in row['sha256'].items()),path
    return paths


def load_operator():
    verify_primitive()
    name='_vspec_proved_two_bank_opaque'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,PRIMITIVE/'opaque.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name]
    assert Path(module.__file__).resolve()==(PRIMITIVE/'opaque.py').resolve()
    assert Path(module.kernel.__file__).resolve()==(PRIMITIVE/'kernel.py').resolve()
    import torch
    return torch.ops.vspec_split_embedding.lookup.default


def validate_bank(bank,group,torch,format_of):
    local=bank.embedding_source
    full=bank.full_embedding
    if (bank.variant!='embedding' or group.world_size!=2 or type(group.rank_in_group) is not int
            or group.rank_in_group not in (0,1) or tuple(local.shape)!=(HALF,WIDTH)
            or tuple(full.shape)!=(2*HALF,WIDTH) or local.dtype!=torch.bfloat16
            or full.dtype!=torch.bfloat16 or local.device!=full.device
            or not local.is_contiguous() or not full.is_contiguous()
            or format_of(local) not in (0,2) or format_of(full) not in (0,2)):
        raise RuntimeError('Two-bank embedding requires pinned immutable native ND BF16 TP2 lookup bank')
    bank.check_sources()
    return local,full,group.rank_in_group


def repack(bank,group):
    import torch
    import torch_npu
    local,full,rank=validate_bank(bank,group,torch,torch_npu.get_npu_format)
    local_pointer=local.data_ptr()
    original_version=local._version
    start=(1-rank)*HALF
    with torch.inference_mode(False):
        remote=torch.empty((HALF,WIDTH),device=full.device,dtype=full.dtype)
        remote.copy_(full[start:start+HALF])
    if (not torch.equal(remote.view(torch.int16),full[start:start+HALF].view(torch.int16))
            or remote.untyped_storage().nbytes()!=HALF*WIDTH*2
            or remote.untyped_storage().data_ptr()==local.untyped_storage().data_ptr()):
        raise RuntimeError('Remote half must be independent exact BF16 storage')
    old_bytes=full.untyped_storage().nbytes()
    if old_bytes!=2*HALF*WIDTH*2:raise RuntimeError('Unexpected full-replica storage prevents budget proof')
    bank._two_bank_local_pointer=local_pointer
    bank._two_bank_local_version=original_version
    bank._two_bank_rank=rank
    bank._two_bank_remote=remote
    bank._two_bank_remote_version=remote._version
    with torch.inference_mode(False):
        bank._two_bank_counter=torch.zeros(8193,device=full.device,dtype=torch.int32)
    bank._two_bank_operator=load_operator()
    bank._two_bank_saved_bytes=old_bytes-remote.untyped_storage().nbytes()
    # Do not rebind any target or local embedding parameter/storage.
    bank.full_embedding=None
    del full
    # Independent ORIGINAL native TP lookup for every checkpoint row, after
    # repacking. This is outside capture/timers and includes remote tokens.
    for start in range(0,2*HALF,4096):
        ids=torch.arange(start,min(start+4096,2*HALF),device=local.device,dtype=torch.int64)
        result=bank._two_bank_operator(ids,local,remote,bank._two_bank_counter,rank,BLOCK)
        expected=bank.original_embedding(ids)
        if not torch.equal(result.view(torch.int16),expected.view(torch.int16)):
            raise RuntimeError('Two-bank full vocabulary differs from original native TP BF16 bits')
    check_bank(bank)
    print('[q35-two-bank-embedding-load] '+json.dumps(dict(
        pid=__import__('os').getpid(),rank=rank,full_vocabulary_native_TP_bits_exact=True,
        full_replica_tensor_removed=bank.full_embedding is None,saved_persistent_bytes=bank._two_bank_saved_bytes,
        actual_remote_storage_bytes=remote.untyped_storage().nbytes(),
        original_local_parameter_pointer_and_version_unchanged=True,
        target_embedding_storage_and_methods_unchanged=True,maximum_fastpath_rows=MAX_ROWS,block=BLOCK)),flush=True)


def check_bank(bank):
    import torch
    bank.check_sources()
    if not torch.compiler.is_compiling():
        if (bank.embedding_source.data_ptr()!=bank._two_bank_local_pointer
                or bank.embedding_source._version!=bank._two_bank_local_version
                or bank._two_bank_remote._version!=bank._two_bank_remote_version
                or bank.full_embedding is not None):
            raise RuntimeError('Immutable two-bank embedding source changed after load')


def lookup_admitted(ids,bank,torch):
    return bool(ids.ndim==1 and ids.dtype in (torch.int32,torch.int64)
                and ids.device==bank.embedding_source.device and ids.is_contiguous()
                and 0<ids.numel()<=MAX_ROWS)


def install_two_bank_embedding():
    if not q35_two_bank_embedding_enabled():return False
    if q35_replicated_inputs()!='embedding':raise RuntimeError('Two-bank layout requires embedding-only private replica')
    verify_primitive();load_operator()
    from .q35_draft_inputs import ReplicatedDraftInputs as cls
    if getattr(cls.__init__,'_q35_two_bank_embedding',False):return False
    original_init,original_lookup=cls.__init__,cls.lookup
    original_metrics,original_install=cls.metrics,cls.install

    @wraps(original_init)
    def initialize(bank,proposer,target_model,group,variant,validate):
        original_init(bank,proposer,target_model,group,variant,validate)
        repack(bank,group)

    @wraps(original_lookup)
    def lookup(bank,ids):
        if not hasattr(bank,'_two_bank_remote'):return original_lookup(bank,ids)
        import torch
        check_bank(bank)
        # The full vLLM compile range MUST use one opaque branch. A Python
        # <=208 guard traced at8192 hardwired native TP into no-guards graphs.
        # Outside the full1..8192 range/nonstandard input keeps native TP.
        if not lookup_admitted(ids,bank,torch):return bank.original_embedding(ids)
        # Preserve the original full-table lookup's explicit int64 index cast.
        result=bank._two_bank_operator(ids.long(),bank.embedding_source,bank._two_bank_remote,
                                      bank._two_bank_counter,bank._two_bank_rank,BLOCK)
        if bank.validate:bank.compare(result,bank.original_embedding(ids),ids.numel(),0)
        if not torch.compiler.is_compiling():bank.captured_embedding.add(ids.numel())
        return result

    @wraps(original_metrics)
    def metrics(bank):
        record=original_metrics(bank)
        if not hasattr(bank,'_two_bank_remote'):return record
        check_bank(bank)
        record['on_mode_replay_counts_not_instrumented']=False
        record['two_bank_embedding']=dict(
            rank=bank._two_bank_rank,block=BLOCK,maximum_fastpath_rows=MAX_ROWS,
            actual_replay_counts={str(n):v for n,v in enumerate(bank._two_bank_counter.cpu().tolist()) if v},
            original_local_parameter_pointer_and_version_unchanged=True,
            remote_weight_version_unchanged=True,full_replica_tensor_removed=bank.full_embedding is None,
            saved_persistent_bytes=bank._two_bank_saved_bytes,
            actual_remote_storage_bytes=bank._two_bank_remote.untyped_storage().nbytes(),
            target_embedding_storage_and_methods_unchanged=True,
            large_input_original_native_TP_fallback=True,
            stats_read_after_timers_only=True)
        return record

    @wraps(original_install)
    def install(bank,manager):
        result=original_install(bank,manager)
        if not hasattr(bank,'_two_bank_remote'):return result
        old_reset=manager.reset_stats
        def reset(_manager):
            record=old_reset();bank._two_bank_counter.zero_();return record
        manager.reset_stats=MethodType(reset,manager)
        return result

    initialize._q35_two_bank_embedding=True
    cls.__init__,cls.lookup,cls.metrics,cls.install=initialize,lookup,metrics,install
    return True
