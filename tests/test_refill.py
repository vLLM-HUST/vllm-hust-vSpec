from vllm_hust_vspec.adaptive.refill import OnlineRefillController


def test_refill_controller_defers_irreversible_admission_probe() -> None:
    controller = OnlineRefillController(max_consecutive_holds=2)
    first = controller.choose(
        running_count=11,
        target_count=16,
        waiting_count=20,
        gamma=4,
        current_goodput=1.0,
        target_goodput=1.0,
    )
    assert not first.admit
    assert first.reason == "measure_hold"

    controller.observe_step(
        running_count=11,
        gamma=4,
        action="hold",
        useful_tokens=100,
        latency_ms=100.0,
    )
    second = controller.choose(
        running_count=11,
        target_count=16,
        waiting_count=20,
        gamma=4,
        current_goodput=1.0,
        target_goodput=1.0,
    )
    assert not second.admit
    assert second.reason == "defer_unmeasured_admit"

    third = controller.choose(
        running_count=11,
        target_count=16,
        waiting_count=20,
        gamma=4,
        current_goodput=1.0,
        target_goodput=1.4,
    )
    assert third.admit
    assert third.reason == "starvation_guard"


def test_refill_controller_uses_destination_gain_for_first_admission() -> None:
    controller = OnlineRefillController(max_consecutive_holds=8)
    controller.observe_step(
        running_count=11,
        gamma=4,
        action="hold",
        useful_tokens=100,
        latency_ms=100.0,
    )

    decision = controller.choose(
        running_count=11,
        target_count=16,
        waiting_count=20,
        gamma=4,
        current_goodput=1.0,
        target_goodput=1.4,
    )

    assert decision.admit
    assert decision.reason == "optimistic_target_gain"


def test_refill_controller_uses_measured_reward_and_future_goodput() -> None:
    controller = OnlineRefillController(
        exploration=0.0,
        future_discount=1.0,
        hysteresis=0.0,
        max_consecutive_holds=3,
    )
    controller.observe_step(
        running_count=7,
        gamma=2,
        action="hold",
        useful_tokens=70,
        latency_ms=100.0,
    )
    controller.observe_step(
        running_count=7,
        gamma=2,
        action="hold",
        useful_tokens=70,
        latency_ms=100.0,
    )
    controller.observe_step(
        running_count=7,
        gamma=2,
        action="admit",
        useful_tokens=40,
        latency_ms=100.0,
    )
    decision = controller.choose(
        running_count=7,
        target_count=16,
        waiting_count=20,
        gamma=2,
        current_goodput=0.7,
        target_goodput=1.5,
    )
    assert decision.admit
    assert decision.reason == "goodput_admit"


def test_refill_controller_uses_exact_target_state_continuation_value() -> None:
    controller = OnlineRefillController(
        exploration=0.0,
        future_discount=1.0,
        hysteresis=0.0,
        max_consecutive_holds=8,
    )
    controller.observe_step(
        running_count=15,
        gamma=5,
        action="hold",
        useful_tokens=30,
        latency_ms=100.0,
    )
    controller.observe_step(
        running_count=15,
        gamma=5,
        action="admit",
        useful_tokens=10,
        latency_ms=100.0,
    )
    controller.observe_step(
        running_count=16,
        gamma=5,
        action="hold",
        useful_tokens=100,
        latency_ms=100.0,
    )

    decision = controller.choose(
        running_count=15,
        target_count=16,
        waiting_count=1,
        gamma=5,
        # The gamma controller pools both states into one B16 bucket.
        current_goodput=0.8,
        target_goodput=0.8,
    )

    assert decision.admit
    assert decision.reason == "goodput_admit"


def test_refill_controller_bounds_holds_to_prevent_starvation() -> None:
    controller = OnlineRefillController(max_consecutive_holds=1)
    first = controller.choose(
        running_count=3,
        target_count=4,
        waiting_count=1,
        gamma=2,
        current_goodput=0.0,
        target_goodput=0.0,
    )
    second = controller.choose(
        running_count=3,
        target_count=4,
        waiting_count=1,
        gamma=2,
        current_goodput=0.0,
        target_goodput=0.0,
    )
    assert not first.admit
    assert second.admit
    assert second.reason == "starvation_guard"


def test_zero_max_holds_uses_queue_driven_gate() -> None:
    controller = OnlineRefillController(max_consecutive_holds=0)
    for _ in range(4):
        decision = controller.choose(
            running_count=15,
            target_count=16,
            waiting_count=1,
            gamma=5,
            current_goodput=1.0,
            target_goodput=1.0,
        )
        assert not decision.admit
        if decision.reason == "measure_hold":
            controller.observe_step(
                running_count=15,
                gamma=5,
                action="hold",
                useful_tokens=100,
                latency_ms=100.0,
            )
        else:
            assert decision.reason == "defer_unmeasured_admit"


def test_cohort_gate_refills_only_after_active_requests_drain() -> None:
    controller = OnlineRefillController(cohort_gated=True)
    active = controller.choose(
        running_count=7,
        target_count=16,
        waiting_count=9,
        gamma=5,
        current_goodput=1.0,
        target_goodput=1.4,
    )
    drained = controller.choose(
        running_count=0,
        target_count=16,
        waiting_count=16,
        gamma=5,
        current_goodput=0.0,
        target_goodput=1.4,
    )

    assert not active.admit
    assert active.reason == "cohort_gate"
    assert drained.admit
    assert drained.reason == "empty_engine"
