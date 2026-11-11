"""CPU-only checks for importing a frozen judged-dataset case list."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples/qwen35_three_datasets/select_cases.py"


def _module():
    spec = spec_from_file_location("select_cases_existing", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_judged_selection_preserves_plan_order(tmp_path):
    module = _module()
    plan = tmp_path / "plan"
    plan.mkdir()
    tasks = [{"case_id": case_id, "track": "gold"} for case_id in ("a", "b", "c")]
    (plan / "tasks.json").write_text(json.dumps(tasks))
    (plan / "contract.json").write_text(json.dumps({
        "tasks_sha256": module.sha256(plan / "tasks.json")
    }))
    source = tmp_path / "old-selection.json"
    source.write_text(json.dumps({
        "case_ids": ["c", "a"],
        "official_plan_contract_sha256": module.sha256(plan / "contract.json"),
    }))

    output = tmp_path / "new-selection.json"
    result = module.select_judged("hle-verified", plan, output, 2, "imported", source)

    assert result["case_ids"] == ["a", "c"]
    assert result["imported_selection_sha256"] == module.sha256(source)
    assert json.loads(output.read_text())["source_plan_contract_sha256"] == module.sha256(
        plan / "contract.json"
    )


def test_existing_judged_selection_rejects_foreign_plan(tmp_path):
    module = _module()
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "tasks.json").write_text('[{"case_id":"a","track":"gold"}]')
    (plan / "contract.json").write_text(json.dumps({
        "tasks_sha256": module.sha256(plan / "tasks.json")
    }))
    source = tmp_path / "old-selection.json"
    source.write_text(json.dumps({
        "case_ids": ["a"], "official_plan_contract_sha256": "not-this-plan",
    }))

    with pytest.raises(ValueError, match="does not belong"):
        module.select_judged("hle-verified", plan, tmp_path / "new.json", 1,
                             "imported", source)
