"""Read actual installed objects; preserve truthful coordinator/receiver registries."""
import hashlib
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(static,root,rank,pid):
    root=Path(root).resolve()
    assert root != root.parent and root.is_dir() and static._use_process_isolation
    assert rank in (0,1) and type(pid) is int and pid>0
    assert static._current_install_path is not None
    path=Path(static._current_install_path).resolve()
    assert path.is_dir() and path.parent==root/'static_kernel_install'
    packages=sorted(static._installed_run_pkgs)
    owner=static._owns_install_path
    if rank==0:
        assert owner==pid and packages
        assert path.name.endswith('_'+str(pid))
    else:
        assert owner is None and not packages
    objects=sorted(path.rglob('*.o'));assert objects
    binaries={};metadata={}
    for p in objects:
        assert p.resolve().is_relative_to(path) and p.is_file() and p.stat().st_size>0
        info=p.with_suffix('.json')
        assert info.resolve().is_relative_to(path) and info.is_file() and info.stat().st_size>0
        binaries[str(p.relative_to(path))]=sha(p)
        metadata[str(info.relative_to(path))]=sha(info)
    return dict(pid=pid,rank=rank,process_isolation=True,private_install_path=str(path),
        install_owner_pid=owner,installed_package_hashes=packages,
        actual_installed_device_binary_sha256=binaries,actual_installed_binary_metadata_sha256=metadata,
        coordinator_installs_once_receiver_registry_remains_truthfully_empty=True,
        offline_compiler_jobs=int(static._compute_opc_compile_jobs()),
        Python_metadata_only_no_tensor_reads_writes_or_resets=True)
def verify_pair(records):
    assert len(records)==2 and {r['rank'] for r in records}=={0,1}
    a=next(r for r in records if r['rank']==0);b=next(r for r in records if r['rank']==1)
    assert a['pid']!=b['pid'] and a['install_owner_pid']==a['pid'] and b['install_owner_pid'] is None
    assert a['installed_package_hashes'] and not b['installed_package_hashes']
    assert a['private_install_path']==b['private_install_path']
    for k in ('actual_installed_device_binary_sha256','actual_installed_binary_metadata_sha256'):
        assert a[k] and a[k]==b[k]
    assert all(r['process_isolation'] and r['offline_compiler_jobs']==8 for r in records)
    return True
