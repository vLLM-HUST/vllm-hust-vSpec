"""Compile the shared hot static shapes without changing graph capture or FX math."""

from copy import deepcopy
from functools import wraps

HOT = (14, 42, 70, 98, 126)
CALLS = []
_RANGE_KEY = "_vllm_aclnn_static_kernel_sym_range"
_FIXED = {
    "force_eager": True,
    "inplace_pass": False,
    "clone_input": False,
    "clone_output": False,
    "static_kernel_compile": True,
}


def select(kwargs):
    if not isinstance(kwargs, dict) or set(kwargs) != {"options"}:
        raise ValueError("Unexpected compiler options envelope")
    original = kwargs["options"]
    if not isinstance(original, dict) or set(original) != {*_FIXED, _RANGE_KEY}:
        raise ValueError("Unexpected static compiler options")
    if any(original[key] is not value for key, value in _FIXED.items()):
        raise ValueError("Shared static compiler switches differ")
    existing = original[_RANGE_KEY]
    if (
        not isinstance(existing, list)
        or any(type(value) is not int or not 1 <= value <= 208 for value in existing)
        or existing != sorted(set(existing))
        or 14 not in existing
        or 70 not in existing
    ):
        raise ValueError("Unexpected static shape range")
    result = deepcopy(kwargs)
    result["options"][_RANGE_KEY] = [value for value in HOT if value in existing]
    return result


def install(module):
    original = module._process_kwargs_options
    if getattr(original, "_vspec_hot_static_metadata", False):
        raise RuntimeError("Static shape selector is already installed")

    @wraps(original)
    def configure(config, kwargs):
        before = deepcopy(kwargs)
        selected = select(kwargs)
        result = original(config, selected)
        if kwargs != before:
            raise RuntimeError("Static compiler mutated the supplied options")
        CALLS.append({"original_options": before, "selected_options": deepcopy(selected)})
        return result

    configure._vspec_hot_static_metadata = True
    module._process_kwargs_options = configure
    return original
