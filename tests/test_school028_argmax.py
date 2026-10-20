import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_hust_vspec import school028


def proposer(**changes):
    fields = dict(method="mtp", num_speculative_tokens=2, model=SimpleNamespace())
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_default_off(monkeypatch):
    monkeypatch.delenv("HUST_VSPEC_SCHOOL028_ARGMAX", raising=False)
    monkeypatch.setattr(school028.importlib.util, "find_spec", lambda _: pytest.fail("host import"))
    school028.register()


def test_greedy_mtp2_only():
    school028.validate_proposer(proposer(), SimpleNamespace(all_greedy=True))
    for value in (
        proposer(method="eagle3"),
        proposer(num_speculative_tokens=1),
        proposer(_enable_probabilistic_draft_probs=True),
        proposer(model=SimpleNamespace(draft_id_to_target_id=[0])),
    ):
        with pytest.raises(ValueError):
            school028.validate_proposer(value, None)
    with pytest.raises(ValueError):
        school028.validate_proposer(proposer(), SimpleNamespace(all_greedy=False))


def test_manifest():
    from vllm_hust_ext.manifest import load_manifest

    path = Path(school028.__file__).with_name("vllm-hust-extension-v0.3.json")
    data = json.loads(path.read_text())
    assert data["host"]["version_range"] == "==0.28.1.post1.dev143"
    assert data["resource_claims"][0]["mode"] == "exclusive"
    load_manifest(path)


def test_head_event_preserves_input_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("HUST_VSPEC_SCHOOL_EVENT_DIR", str(tmp_path))
    school028.event("head_equivalence", rank=0, rows=24, kind="zeros", exact=True)
    row = json.loads(next(tmp_path.glob("vspec-*.jsonl")).read_text())
    assert row["event"] == "head_equivalence"
    assert row["kind"] == "zeros" and row["exact"]
