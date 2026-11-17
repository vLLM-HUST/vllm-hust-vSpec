"""Read actual process-isolated static kernel install metadata after timing."""

import hashlib
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(static, root, rank, pid):
    root = Path(root).resolve(strict=True)
    if root == root.parent or not root.is_dir() or not static._use_process_isolation:
        raise RuntimeError("Static kernels require a private process-isolated directory")
    if rank not in (0, 1) or type(pid) is not int or pid <= 0:
        raise ValueError("Expected a two-rank static install")
    current = static._current_install_path
    if current is None:
        raise RuntimeError("Static kernel install path is unavailable")
    path = Path(current).resolve(strict=True)
    if not path.is_dir() or path.parent != root / "static_kernel_install":
        raise RuntimeError("Static kernels were installed outside the private run")
    packages = sorted(static._installed_run_pkgs)
    owner = static._owns_install_path
    if rank == 0:
        if owner != pid or not packages or not path.name.endswith(f"_{pid}"):
            raise RuntimeError("Coordinator did not own the static kernel install")
    elif owner is not None or packages:
        raise RuntimeError("Receiver unexpectedly installed static kernels")

    objects = sorted(path.rglob("*.o"))
    if not objects:
        raise RuntimeError("No installed static device binaries were found")
    binaries = {}
    metadata = {}
    for binary in objects:
        info = binary.with_suffix(".json")
        for item in (binary, info):
            if (
                not item.is_file()
                or not item.resolve(strict=True).is_relative_to(path)
                or item.stat().st_size <= 0
            ):
                raise RuntimeError("Installed static binary or metadata is incomplete")
        binaries[str(binary.relative_to(path))] = _sha256(binary)
        metadata[str(info.relative_to(path))] = _sha256(info)

    return {
        "pid": pid,
        "rank": rank,
        "process_isolation": True,
        "private_install_path": str(path),
        "install_owner_pid": owner,
        "installed_package_hashes": packages,
        "actual_installed_device_binary_sha256": binaries,
        "actual_installed_binary_metadata_sha256": metadata,
        "coordinator_installs_once_receiver_registry_remains_truthfully_empty": True,
        "offline_compiler_jobs": int(static._compute_opc_compile_jobs()),
        "Python_metadata_only_no_tensor_reads_writes_or_resets": True,
    }


def verify_pair(records):
    if len(records) != 2 or {row["rank"] for row in records} != {0, 1}:
        raise ValueError("Expected one coordinator and one receiver")
    coordinator = next(row for row in records if row["rank"] == 0)
    receiver = next(row for row in records if row["rank"] == 1)
    if (
        coordinator["pid"] == receiver["pid"]
        or coordinator["install_owner_pid"] != coordinator["pid"]
        or receiver["install_owner_pid"] is not None
        or not coordinator["installed_package_hashes"]
        or receiver["installed_package_hashes"]
        or coordinator["private_install_path"] != receiver["private_install_path"]
        or any(not row["process_isolation"] or row["offline_compiler_jobs"] != 8 for row in records)
    ):
        raise ValueError("Static install registries do not match the two-rank contract")
    for key in (
        "actual_installed_device_binary_sha256",
        "actual_installed_binary_metadata_sha256",
    ):
        if not coordinator[key] or coordinator[key] != receiver[key]:
            raise ValueError("Static install binaries differ between workers")
    return True
