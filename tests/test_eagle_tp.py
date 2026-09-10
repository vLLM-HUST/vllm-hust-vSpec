from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from vllm_hust_vspec.backends.eagle_tp import (
    _capture_draft_tp_group,
    _wrap_method_in_draft_tp_group,
)


def _proposer(*, target_tp: int, draft_tp: int):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=target_tp),
        ),
        speculative_config=SimpleNamespace(
            draft_tensor_parallel_size=draft_tp,
        ),
    )


def test_capture_replicated_draft_tp_group() -> None:
    draft_group = object()
    proposer = _proposer(target_tp=2, draft_tp=1)

    @contextmanager
    def group_context(group):
        yield group

    proposer.tp_group_context = group_context(draft_group)
    _capture_draft_tp_group(proposer)

    assert proposer._vspec_draft_tp_group is draft_group


def test_capture_same_tp_uses_no_override() -> None:
    proposer = _proposer(target_tp=2, draft_tp=2)
    proposer.tp_group_context = object()

    _capture_draft_tp_group(proposer)

    assert proposer._vspec_draft_tp_group is None


def test_wrapped_method_uses_draft_tp_group() -> None:
    events: list[tuple[str, object]] = []
    draft_group = object()

    @contextmanager
    def patch_tensor_parallel_group(group):
        events.append(("enter", group))
        try:
            yield
        finally:
            events.append(("exit", group))

    class Proposer:
        _vspec_draft_tp_group = draft_group

        def execute(self, value: int) -> int:
            events.append(("execute", self._vspec_draft_tp_group))
            return value + 1

    _wrap_method_in_draft_tp_group(
        Proposer,
        "execute",
        patch_tensor_parallel_group,
    )

    assert Proposer().execute(4) == 5
    assert events == [
        ("enter", draft_group),
        ("execute", draft_group),
        ("exit", draft_group),
    ]


def test_wrapped_method_without_draft_group_is_transparent() -> None:
    entered = False

    @contextmanager
    def patch_tensor_parallel_group(group):
        nonlocal entered
        entered = True
        yield group

    class Proposer:
        _vspec_draft_tp_group = None

        def execute(self) -> str:
            return "target"

    _wrap_method_in_draft_tp_group(
        Proposer,
        "execute",
        patch_tensor_parallel_group,
    )

    assert Proposer().execute() == "target"
    assert entered is False
