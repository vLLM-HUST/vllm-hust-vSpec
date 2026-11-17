from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from vllm_hust_vspec.runtime_installer import EXPORTS, OPERATORS, commands


def test_full_install_plan_builds_and_validates_each_operator(tmp_path: Path) -> None:
    args = Namespace(
        wheel=tmp_path / "vspec.whl",
        plugin_source=None,
        ascend_source=tmp_path / "ascend",
        vllm_source=tmp_path / "vllm",
        cann_root=tmp_path / "cann",
        model_dir=tmp_path / "model",
        private_dir=tmp_path / "private",
        jobs=2,
    )
    plan = commands(args, tmp_path / "site")
    labels = [label for label, _ in plan]

    assert labels[:3] == ["plugin", "backend-source", "backend-wheel"]
    assert "--force-reinstall" in plan[0][1]
    for operator, _, _ in OPERATORS:
        build = labels.index(f"build-{operator}")
        assert labels[build + 1] == f"validate-{operator}"
        assert str(args.private_dir / "components" / operator) in plan[build][1]
    assert labels[-6:] == [
        "validate-gdn-6",
        "validate-gdn-8",
        "validate-mask",
        "validate-prefix-clip",
        "validate-two-bank",
        "validate-inplace-head",
    ]
    assert all(
        component in [name for name, _, _ in OPERATORS]
        or component
        in {"gdn-input-6", "gdn-input-8", "mask", "prefix-clip", "two-bank", "inplace-head"}
        for component in EXPORTS.values()
    )
