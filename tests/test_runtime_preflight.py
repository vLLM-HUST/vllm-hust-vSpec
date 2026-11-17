import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_preflight_checks_every_component_and_stops_on_failure(monkeypatch):
    checked = []

    def check(name):
        def run():
            checked.append(name)
            if name == 'gamma8_rgdr':
                raise RuntimeError('unverified')
        return run

    hooks = ModuleType('vllm_hust_vspec._vendor.experimental_hooks')
    hooks.gamma6_mask_hook = SimpleNamespace(verify=check('gamma6_mask'))
    hooks.gamma6_tail_hook = SimpleNamespace(verify=check('gamma6_tail'))
    hooks.gamma8_input_domain = SimpleNamespace(verify=check('gamma8_input'))
    hooks.gamma8_rgdr_hook = SimpleNamespace(verify=check('gamma8_rgdr'))
    hooks.native_stride_conv = SimpleNamespace(verify_primitive=check('native_stride'),
                                               verify_edge_primitive=check('edge_stride'))
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    fia = ModuleType('vllm_hust_vspec._vendor.runtime.native_fia_host_batch_max8')
    fia.verify_primitive = check('fia')
    monkeypatch.setitem(sys.modules, fia.__name__, fia)
    preflight = importlib.import_module('vllm_hust_vspec._vendor.runtime.preflight')
    with pytest.raises(RuntimeError, match='unverified'):
        preflight.verify_components()
    assert checked == ['native_stride', 'edge_stride', 'fia', 'gamma8_input', 'gamma8_rgdr']
