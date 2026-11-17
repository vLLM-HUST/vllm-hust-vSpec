"""CPU-only bounded payback budget; never admits or commits a GPU transition."""
import math

def probe_budget(selector,ticket,rows):
    window=selector.window;maximum=window+selector.amortization_rounds
    result=dict(required_rounds=maximum,legacy_maximum_rounds=maximum,
        method='unknown_cost_keep_original_budget',gain_source='conservative_hysteresis',
        measured_generation_wall_NOT_modified=True,no_labels_or_imported_profile=True)
    if (ticket is None or ticket.reason not in ('adjacent_higher_trial','adjacent_lower_trial_missing_context')
            or ticket.old!=selector.current_gamma):return result
    cost=selector.switch_cost_ms
    if not rows or not math.isfinite(cost) or cost<=0:return result
    step=sum(r[1] for r in rows)/len(rows)
    if not math.isfinite(step) or step<=0:return result
    gain=max(.01,selector.hysteresis)
    prediction=(selector.cost_gate_last if ticket.new>ticket.old else selector.lower_gate_last)
    context=selector.context
    batch=context[0] if isinstance(context,tuple) and context and type(context[0]) is int else None
    if (isinstance(prediction,dict) and batch is not None
            and prediction.get('feedback_round')==selector.rounds
            and (prediction.get('batch'),prediction.get('old'),prediction.get('new'))==(batch,ticket.old,ticket.new)):
        current=prediction.get('current_goodput',0);future=prediction.get('predicted_goodput',0)
        if current>0 and math.isfinite(current) and math.isfinite(future):
            predicted=future/current-1
            if math.isfinite(predicted) and predicted>gain:
                gain=min(1.,predicted);result['gain_source']='fresh_matching_current_round_prediction'
    # Last switch RPC wall is a live cost estimate, not a strict upper bound.
    # Reserve a round trip and a FULL configured calibration window. Bound the
    # result by the previous80-round rule; all original state/roster/device
    # proofs remain mandatory and actual benchmark wall still includes ALL work.
    ratio=2*cost/(step*gain)
    if not math.isfinite(ratio):return result
    payback=math.ceil(ratio)
    settling=getattr(selector,'steady_settling_observations',0)
    required=min(maximum,max(window+2+settling,window+payback+settling))
    result.update(required_rounds=required,method='live_RPC_cost_and_gain_payback',
        last_measured_switch_RPC_ms=cost,current_mean_round_ms=step,
        estimated_gain_fraction=gain,estimated_roundtrip_cost_ms=2*cost,
        calibration_rounds=window,estimated_payback_rounds=payback,
        post_RPC_estimator_settling_rounds=settling,
        cost_transfer_is_heuristic_NOT_performance_guarantee=True)
    return result
