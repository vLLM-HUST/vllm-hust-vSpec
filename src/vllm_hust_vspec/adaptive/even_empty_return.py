"""Honor ONLY a measured-loser return at the existing empty-cohort boundary.

This creates a CPU policy ticket, never a device admission receipt. The existing
scheduler's authoritative boundary marker and unchanged worker synchronization
remain required. It cannot switch widths inside a live cohort.
"""

def measured_empty_return(selector, *, at_cohort_boundary, fixed_control=False):
    if at_cohort_boundary is not True or fixed_control:
        return None
    target=selector.deferred_measured_return
    if target is None or target==selector.current_gamma:
        return None
    if target not in selector.candidates or abs(target-selector.current_gamma)!=2:
        raise RuntimeError('Deferred measured return must remain adjacent and supported')
    ticket=selector.pending
    if ticket is not None:
        if (ticket.old!=selector.current_gamma or ticket.new!=target
            or ticket.reason not in ('measured_probe_not_better_return',
                'deferred_measured_probe_not_better_return')):
            raise RuntimeError('Cannot replace an unrelated pending transition')
        return ticket
    return selector._request(target,'deferred_measured_probe_not_better_return')
