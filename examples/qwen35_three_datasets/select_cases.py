"""Freeze one deterministic case selection from an installed evaluation client's plan.

This program never downloads data or changes the source plan. The derived plan
or case-ID manifest belongs in a private work directory, not in this repo.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mmlu-pro", "hle-verified", "frontierscience")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def private_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not path.is_absolute() or resolved.is_relative_to(REPO_ROOT):
        raise ValueError("Selection output must be absolute and outside the source repository")
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing selection: {resolved}")
    return resolved


def _stratum(dataset: str, task: dict[str, Any]) -> str:
    if dataset == "mmlu-pro":
        return str(task["category"])
    if dataset == "frontierscience":
        return str(task["track"])
    return str(task.get("track", "gold"))


def choose_ids(
    dataset: str, tasks: list[dict[str, Any]], count: int | None, seed: str
) -> list[str]:
    """Hash-order within each stratum, then round-robin strata; return plan order."""
    if not tasks or not seed:
        raise ValueError("A nonempty plan and seed are required")
    if count is None:
        count = len(tasks)
    if count < 1 or count > len(tasks):
        raise ValueError("Selection count must be between 1 and plan size")
    ids = [task["case_id"] for task in tasks]
    if len(ids) != len(set(ids)) or any(not isinstance(case_id, str) for case_id in ids):
        raise ValueError("Plan contains duplicate or non-string case IDs")
    groups: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        groups[_stratum(dataset, task)].append(task["case_id"])
    for members in groups.values():
        members.sort(key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode()).digest())
    selected: set[str] = set()
    strata = sorted(groups)
    while len(selected) < count:
        for stratum in strata:
            if groups[stratum]:
                selected.add(groups[stratum].pop(0))
                if len(selected) == count:
                    break
    return [case_id for case_id in ids if case_id in selected]


def _selection(dataset: str, plan: Path, tasks: list[dict[str, Any]], ids: list[str], seed: str):
    id_bytes = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    selected = set(ids)
    return {
        "schema_version": 1,
        "dataset": dataset,
        "source_plan_contract_sha256": sha256(plan / "contract.json"),
        "selection_seed": seed,
        "count": len(ids),
        "case_ids": ids,
        "case_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
        "strata": dict(
            sorted(
                Counter(
                    _stratum(dataset, row) for row in tasks if row["case_id"] in selected
                ).items()
            )
        ),
    }


def select_mmlu(plan: Path, output: Path, count: int | None, seed: str) -> dict[str, Any]:
    from hust_eval_mmlu_pro.mmlu_pro import plan as client_plan

    contract, cases = client_plan.load_plan(plan)
    ids = choose_ids("mmlu-pro", cases, count, seed)
    selected = set(ids)
    rows = [case for case in cases if case["case_id"] in selected]
    selection = _selection("mmlu-pro", plan, cases, ids, seed)
    output.mkdir(parents=True, mode=0o700)
    write_json(output / "cases.json", rows)
    write_json(
        output / "tasks.json", [{k: v for k, v in row.items() if k != "prompt"} for row in rows]
    )
    shutil.copyfile(plan / "inventory.jsonl.gz", output / "inventory.jsonl.gz")
    (output / "inventory.jsonl.gz").chmod(0o600)
    write_json(output / "selection.json", selection)
    derived = copy.deepcopy(contract)
    derived["tasks"] = len(rows)
    derived["selection"] = "deterministic stratified subset; see selection.json"
    derived["subset_provenance_sha256"] = sha256(output / "selection.json")
    derived["files_sha256"] = {
        name: sha256(output / name)
        for name in ("cases.json", "tasks.json", "inventory.jsonl.gz", "selection.json")
    }
    write_json(output / "contract.json", derived)
    checked, checked_rows = client_plan.load_plan(output)
    if checked["tasks"] != len(rows) or [row["case_id"] for row in checked_rows] != ids:
        raise RuntimeError("Derived MMLU-Pro plan did not verify")
    return selection


def select_judged(
    dataset: str, plan: Path, output: Path, count: int | None, seed: str,
    existing_selection: Path | None = None,
) -> dict[str, Any]:
    contract = json.loads((plan / "contract.json").read_text(encoding="utf-8"))
    tasks = json.loads((plan / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(tasks, list):
        raise ValueError("Invalid client plan")
    if contract.get("tasks_sha256") != sha256(plan / "tasks.json"):
        raise ValueError("Client plan task hash mismatch")
    if existing_selection is None:
        ids = choose_ids(dataset, tasks, count, seed)
    else:
        source = json.loads(existing_selection.read_text(encoding="utf-8"))
        source_ids = source.get("case_ids") if isinstance(source, dict) else None
        if (not isinstance(source_ids, list) or not source_ids
                or not all(isinstance(case_id, str) for case_id in source_ids)):
            raise ValueError("Existing selection has invalid or duplicate case IDs")
        selected = set(source_ids)
        if len(selected) != len(source_ids):
            raise ValueError("Existing selection has invalid or duplicate case IDs")
        source_plan_hash = (source.get("source_plan_contract_sha256")
                            or source.get("official_plan_contract_sha256"))
        if source_plan_hash != sha256(plan / "contract.json"):
            raise ValueError("Existing selection does not belong to this plan")
        if count is not None and len(source_ids) != count:
            raise ValueError("Existing selection count does not match --count")
        plan_ids = [task["case_id"] for task in tasks]
        if not selected <= set(plan_ids):
            raise ValueError("Existing selection contains IDs absent from this plan")
        ids = [case_id for case_id in plan_ids if case_id in selected]
    selection = _selection(dataset, plan, tasks, ids, seed)
    if existing_selection is not None:
        selection["imported_selection_sha256"] = sha256(existing_selection)
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_json(output, selection)
    return selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, help="omit for the full selected plan")
    parser.add_argument("--existing-selection", type=Path,
                        help="preserve case IDs from a verified prior judged-dataset selection")
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    plan = args.plan.expanduser().resolve(strict=True)
    output = private_output(args.output)
    if args.dataset == "mmlu-pro":
        if args.existing_selection is not None:
            raise ValueError("--existing-selection is only for judged datasets")
        result = select_mmlu(plan, output, args.count, args.seed)
    else:
        source = (args.existing_selection.expanduser().resolve(strict=True)
                  if args.existing_selection is not None else None)
        result = select_judged(args.dataset, plan, output, args.count, args.seed, source)
    print(json.dumps({key: result[key] for key in ("dataset", "count", "case_ids_sha256")}))


if __name__ == "__main__":
    main()
