"""The archived issue-18 policy remains available in the release wheel."""

from vllm_hust_vspec.adaptive.legacy_even_online import EvenRoundOnlineController
from vllm_hust_vspec.adaptive.legacy_even_round import EvenRoundSelector
from vllm_hust_vspec.adaptive import even_runtime


def test_archived_policy_starts_at_eight_and_keeps_even_candidates():
    selector = EvenRoundSelector()
    assert selector.current_gamma == 8
    assert selector.candidates == (2, 4, 6, 8)
    proposal = None
    for _ in range(selector.window):
        proposal = selector.observe(
            gamma=8, accepted=[8], committed=[9], latency_ms=1, context=(1, 0)
        )
    assert (proposal.old, proposal.new) == (8, 6)
    selector.commit(proposal, transition_cost_ms=0)
    assert selector.current_gamma == 6
    for _ in range(selector.window):
        proposal = selector.observe(
            gamma=6, accepted=[6], committed=[7], latency_ms=2, context=(1, 0)
        )
    assert (proposal.old, proposal.new) == (6, 8)
    assert proposal.reason == "measured_probe_not_better_return"


def test_archived_controller_uses_archived_selector():
    assert EvenRoundOnlineController.__module__.endswith("legacy_even_online")


def test_archived_profile_preserves_original_transition_cost(monkeypatch):
    monkeypatch.setenv("HUST_VSPEC_EVEN_POLICY_PROFILE", "archived-20261103")
    phase = even_runtime.EvenPhase(object())
    phase.pending_since = 10.0
    phase.admission_rpc_ms = 3.0
    monkeypatch.setattr(even_runtime.time, "monotonic", lambda: 10.125)
    assert phase.transition_cost_ms() == 125.0

    monkeypatch.setenv("HUST_VSPEC_EVEN_POLICY_PROFILE", "current")
    current = even_runtime.EvenPhase(object())
    current.pending_since = 10.0
    current.admission_rpc_ms = 3.0
    assert current.transition_cost_ms() == 3.0
