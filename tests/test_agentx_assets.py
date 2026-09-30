from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _load(name: str) -> dict[str, object]:
    value = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_agentx_target_templates_preserve_frontier_contract() -> None:
    baseline = _load("agentx-qwen35-frontier-target-only.example.json")
    mtp2 = _load("agentx-qwen35-frontier-mtp2.example.json")

    for target in (baseline, mtp2):
        assert target["max_model_len"] == 262144
        assert target["host_kv_budget_gib"] == 0
        assert target["hardware"] == {
            "accelerator": "Ascend 910B2",
            "device_count": 2,
            "allocated_host_dram_gib": 0,
        }

    assert baseline["speculative_decoding"] == {"enabled": False}
    speculation = mtp2["speculative_decoding"]
    assert isinstance(speculation, dict)
    assert speculation["enabled"] is True
    server = speculation["server_configuration"]
    assert isinstance(server, dict)
    assert server["method"] == "mtp"
    assert server["num_speculative_tokens"] == 2
    assert server["rejection_sample_method"] == "synthetic"
    assert speculation["forced_acceptance_length"] == server["synthetic_acceptance_length"]
    assert "REPLACE_WITH_" in json.dumps(mtp2)
