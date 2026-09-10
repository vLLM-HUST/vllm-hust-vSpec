from __future__ import annotations

import subprocess
import sys
from importlib import resources
from importlib.metadata import entry_points, version
from pathlib import Path

from vllm_hust_ext.discovery import discover_bundles
from vllm_hust_ext.manifest import load_manifest

import vllm_hust_vspec
from vllm_hust_vspec import manifests

EXTENSION_ID = "org.vllm-hust.vspec"
MANIFEST_NAME = "vllm-hust-extension-v0.2.json"


def manifest_path() -> Path:
    return Path(str(resources.files(manifests).joinpath(MANIFEST_NAME)))


def test_manifest_matches_registration_boundary() -> None:
    manifest = load_manifest(manifest_path())

    assert manifest.bundle_id == EXTENSION_ID
    assert manifest.bundle_version == vllm_hust_vspec.__version__
    assert manifest.kind == "in_process_plugin"
    assert manifest.host.provider == "vllm"
    assert manifest.runtime.isolation == "trusted_in_process"
    assert manifest.lifecycle_owner == "vllm"
    assert dict(manifest.activation.environment) == {"HUST_VSPEC_ENABLED": "1"}
    assert version("vllm-hust-vspec") == vllm_hust_vspec.__version__
    assert manifest_path().parent.name == "manifests"
    assert not manifest_path().parent.parent.joinpath(MANIFEST_NAME).exists()

    bundle_registrations = entry_points(group="vllm_hust.extension_bundles")
    assert any(
        item.name == EXTENSION_ID and item.value == "vllm_hust_vspec.manifests"
        for item in bundle_registrations
    )

    plugin_registrations = entry_points(group="vllm.general_plugins")
    assert any(
        item.name == "vspec" and item.value == "vllm_hust_vspec:register"
        for item in plugin_registrations
    )


def test_extension_manager_discovers_static_bundle() -> None:
    bundle = discover_bundles((EXTENSION_ID,))[0]

    assert bundle.bundle_id == EXTENSION_ID
    assert bundle.distribution_name == "vllm-hust-vspec"
    assert bundle.manifest_path.name == MANIFEST_NAME
    assert bundle.manifest_path.parent.name == "manifests"
    assert bundle.manifest.kind == "in_process_plugin"


def test_static_discovery_does_not_import_implementation() -> None:
    script = """
import sys
from vllm_hust_ext.discovery import discover_bundles

assert "vllm_hust_vspec" not in sys.modules
bundle = discover_bundles(("org.vllm-hust.vspec",))[0]
assert bundle.bundle_id == "org.vllm-hust.vspec"
assert "vllm_hust_vspec" not in sys.modules
assert "vllm_hust_vspec.manifests" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)
