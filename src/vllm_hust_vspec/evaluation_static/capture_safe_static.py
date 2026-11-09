"""Defer static compilation while an NPU graph is being captured."""

from functools import wraps


def guarded(original, is_capturing, statistics):
    if getattr(original, "_vspec_capture_safe_static", False):
        raise RuntimeError("Static compilation capture guard is already installed")

    @wraps(original)
    def compile_when_safe(self, *args, **kwargs):
        active = is_capturing()
        if type(active) is not bool:
            raise TypeError("NPU capture state must be a bool")
        if active:
            statistics["static_compilation_deferred_inside_capture"] += 1
            # AclGraph.__call__ invokes its original FX eager path after this
            # returns. Leave compiled_shape untouched so a later safe call can
            # compile the same shape.
            return None
        statistics["original_eager_compile_method_calls"] += 1
        return original(self, *args, **kwargs)

    compile_when_safe._vspec_capture_safe_static = True
    return compile_when_safe


def install(acl_graph, is_capturing):
    cls = acl_graph.AclGraph
    original = cls.run_eagerly_compile
    statistics = {
        "static_compilation_deferred_inside_capture": 0,
        "original_eager_compile_method_calls": 0,
    }
    cls.run_eagerly_compile = guarded(original, is_capturing, statistics)
    return original, statistics
