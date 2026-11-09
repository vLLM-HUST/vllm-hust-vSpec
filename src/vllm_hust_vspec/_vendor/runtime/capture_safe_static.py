"""Defer unsafe optimization work, never change FX math or suppress errors."""
from functools import wraps
def guarded(original,is_capturing,statistics):
    assert not getattr(original,'_vspec_capture_safe_static',False)
    @wraps(original)
    def compile_when_safe(self,*args,**kwargs):
        active=is_capturing()
        assert type(active) is bool
        if active:
            statistics['static_compilation_deferred_inside_capture']+=1
            # Do not call FX, synchronize, install packages, or mark compiled_shape.
            # Original AclGraph.__call__ then invokes its original FX exactly once.
            return None
        statistics['original_eager_compile_method_calls']+=1
        return original(self,*args,**kwargs)
    compile_when_safe._vspec_capture_safe_static=True
    return compile_when_safe
def install(acl_graph,is_capturing):
    cls=acl_graph.AclGraph;original=cls.run_eagerly_compile
    statistics=dict(static_compilation_deferred_inside_capture=0,original_eager_compile_method_calls=0)
    cls.run_eagerly_compile=guarded(original,is_capturing,statistics)
    return original,statistics
