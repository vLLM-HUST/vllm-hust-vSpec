"""The initial width changes without changing the adjacent online policy."""

from unittest.mock import patch

import pytest

from vllm_hust_vspec.adaptive.even_round import (
    EvenRoundSelector,
    initial_even_gamma_from_env,
)
from vllm_hust_vspec.adaptive.runtime import _online_initial_gamma


def test_new_engine_starts_at_eight_and_can_return_after_lower_trial():
    assert initial_even_gamma_from_env({}) == 8
    with patch.dict('os.environ', {'HUST_VSPEC_EVEN_ROUND_ENABLE': '1'}, clear=False):
        assert _online_initial_gamma('mtp', 2, 8) == 8

    selector = EvenRoundSelector(initial_gamma=8)
    assert selector.summary()['initial_gamma'] == selector.current_gamma == 8
    assert selector.bootstrap_committed_widths == {8}
    context = (1, 0, ('request',))
    for _ in range(selector.window):
        proposal = selector.observe(gamma=8, accepted=[8], committed=[9],
                                    latency_ms=1, context=context)
    assert (proposal.old, proposal.new) == (8, 6)
    selector.commit(proposal, transition_cost_ms=0)
    for _ in range(selector.window):
        return_ticket = selector.observe(gamma=6, accepted=[6], committed=[7],
                                         latency_ms=2, context=context)
    assert (return_ticket.old, return_ticket.new) == (6, 8)
    assert return_ticket.reason == 'measured_probe_not_better_return'


def test_explicit_legacy_initial_arm_and_invalid_values():
    assert initial_even_gamma_from_env({'HUST_VSPEC_EVEN_INITIAL_GAMMA': '2'}) == 2
    with patch.dict('os.environ', {
        'HUST_VSPEC_EVEN_ROUND_ENABLE': '1',
        'HUST_VSPEC_EVEN_INITIAL_GAMMA': '2',
    }, clear=False):
        assert _online_initial_gamma('mtp', 2, 8) == 2
    selector = EvenRoundSelector(initial_gamma=2)
    assert selector.summary()['initial_gamma'] == selector.current_gamma == 2
    with pytest.raises(ValueError, match='HUST_VSPEC_EVEN_INITIAL_GAMMA'):
        initial_even_gamma_from_env({'HUST_VSPEC_EVEN_INITIAL_GAMMA': '3'})
    with pytest.raises(ValueError, match='Initial gamma'):
        EvenRoundSelector(maximum=6, initial_gamma=8)
