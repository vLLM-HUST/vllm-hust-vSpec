"""Materialize the official SPEED-Bench qualitative coding subset."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from datasets import load_dataset

DATASET_REVISION = "454f88454792dfa3ccfd7ef15fff248efde44cd1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-prepare", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = runpy.run_path(str(args.official_prepare))
    dataset = load_dataset(
        "nvidia/SPEED-Bench",
        "qualitative",
        split="test",
        revision=DATASET_REVISION,
    )
    dataset = dataset.filter(lambda row: row["category"] == "coding")
    if len(dataset) != 80:
        raise RuntimeError(f"expected 80 coding prompts, got {len(dataset)}")
    dataset = official["_resolve_external_data"](dataset, "qualitative")
    placeholder = official["TURNS_PLACEHOLDER"]
    unresolved = [row["src_id"] for row in dataset if row["turns"][0].startswith(placeholder)]
    if unresolved:
        raise RuntimeError(f"unresolved source references: {unresolved}")
    dataset = dataset.map(
        lambda row: {
            "messages": [{"role": "user", "content": turn} for turn in row["turns"]]
        },
        remove_columns=["turns"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "qualitative.jsonl"
    dataset.to_json(output)
    print(f"Wrote {len(dataset)} official coding prompts to {output}")
    print(f"SPEED-Bench revision: {DATASET_REVISION}")


if __name__ == "__main__":
    main()
