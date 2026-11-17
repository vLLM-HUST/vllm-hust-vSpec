from functools import wraps
from types import SimpleNamespace

from vllm_hust_vspec.adaptive.prefix_bridge_runtime import basic_ready, install_core_step


def test_prefix_bridge_arms_only_uniform_old_width():
    request = SimpleNamespace(
        use_structured_output=False,
        num_computed_tokens=20,
        num_prompt_tokens=10,
        max_tokens=100,
        num_output_tokens=20,
        spec_token_ids=[1] * 8,
    )
    scheduler = SimpleNamespace(
        running=[request], waiting=[], skipped_waiting=[],
        num_waiting_for_streaming_input=0,
    )
    ticket = SimpleNamespace(old=8, new=6)
    controller = SimpleNamespace(current_gamma=8)
    assert basic_ready(scheduler, ticket, controller)
    request.spec_token_ids = [1] * 6
    assert not basic_ready(scheduler, ticket, controller)


def test_armed_prefix_bridge_closes_cohort_for_scheduler_step(monkeypatch):
    from vllm.v1.engine.core import EngineCore

    observed = []

    def native_step(core):
        observed.append(core.scheduler.max_num_running_reqs)
        return "done"

    @wraps(native_step)
    def prior_even_step(core):
        return native_step(core)

    prior_even_step._vspec_even_round = True
    monkeypatch.setattr(EngineCore, "step_with_batch_queue", prior_even_step)
    install_core_step()

    ticket = SimpleNamespace(old=8, new=6)
    controller = SimpleNamespace(
        fixed4_ablation=False, current_gamma=8,
        even=SimpleNamespace(pending=ticket),
    )
    scheduler = SimpleNamespace(
        running=[object(), object()], max_num_running_reqs=16,
        _vspec_adaptive_controller=controller,
    )
    phase = SimpleNamespace(
        scheduler=scheduler,
        _private_prefix_clip_ticket=ticket,
        check_thread=lambda: None,
        requests=lambda: scheduler.running,
    )
    scheduler._vspec_even_phase = phase
    core = SimpleNamespace(scheduler=scheduler, batch_queue=[])

    assert EngineCore.step_with_batch_queue(core) == "done"
    assert observed == [2]
    assert scheduler.max_num_running_reqs == 16
    assert phase._private_prefix_clip_ticket is ticket
