"""Fail closed before the experimental worker installs any runtime hooks."""


def verify_components():
    from ..experimental_hooks import (gamma6_mask_hook, gamma6_tail_hook,
                                      gamma8_input_domain, gamma8_rgdr_hook,
                                      native_stride_conv)
    from . import native_fia_host_batch_max8

    native_stride_conv.verify_primitive()
    native_stride_conv.verify_edge_primitive()
    native_fia_host_batch_max8.verify_primitive()
    gamma8_input_domain.verify()
    gamma8_rgdr_hook.verify()
    gamma6_tail_hook.verify()
    gamma6_mask_hook.verify()
