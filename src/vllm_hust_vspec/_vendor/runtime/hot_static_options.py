"""Only restrict static compilation, never prune graph capture or change FX math."""
from copy import deepcopy
from functools import wraps
import os

_AVAILABLE_HOT=(14,42,70,98,126)
_hot_text=os.environ.get('HUST_VSPEC_HOT_STATIC_SHAPES','70')
if not _hot_text or any(not part.isdecimal() for part in _hot_text.split(',')):
    raise RuntimeError('Invalid hot static shape selection')
HOT=tuple(int(part) for part in _hot_text.split(','))
if HOT!=tuple(sorted(set(HOT))) or 70 not in HOT or any(v not in _AVAILABLE_HOT for v in HOT):
    raise RuntimeError('Hot static shapes must be an ordered subset containing 70')
CALLS=[]
def select(kwargs):
    assert set(kwargs)=={'options'}
    original=kwargs['options']
    fixed=dict(force_eager=True,inplace_pass=False,clone_input=False,clone_output=False,static_kernel_compile=True)
    assert all(original.get(k) is v for k,v in fixed.items())
    key='_vllm_aclnn_static_kernel_sym_range'
    assert set(original)==set(fixed)|{key}
    existing=original[key]
    assert isinstance(existing,list) and existing==sorted(set(existing))
    assert all(type(v) is int and 1<=v<=208 for v in existing)
    assert 14 in existing and 70 in existing
    chosen=[v for v in HOT if v in existing]
    result=deepcopy(kwargs);result['options'][key]=chosen
    restored=deepcopy(result);restored['options'][key]=deepcopy(existing)
    assert restored==kwargs
    return result
def install(module):
    original=module._process_kwargs_options
    assert not getattr(original,'_vspec_hot_static_metadata',False)
    @wraps(original)
    def configure(config,kwargs):
        before=deepcopy(kwargs);selected=select(kwargs)
        result=original(config,selected)
        assert kwargs==before
        CALLS.append(dict(original_options=before,selected_options=deepcopy(selected)))
        return result
    configure._vspec_hot_static_metadata=True
    module._process_kwargs_options=configure
    return original
