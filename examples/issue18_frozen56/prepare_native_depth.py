"""Prepare a fresh copy of the public Issue #18 native capsule for MTP4/6/8.

This changes only the MTP depth and, when requested, extends the original
48-entry graph-capture list to cover a 16-request MTP8 decode batch. Device
selection remains the caller's responsibility; no card number is embedded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ORIGINAL_CAPTURE_SIZES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    18, 20, 21, 24, 25, 27, 28, 30, 33, 35, 36, 39, 40, 42,
    45, 48, 49, 50, 55, 56, 60, 63, 65, 70, 75, 77, 80, 84,
    91, 98, 105, 112,
]
MTP8_ADDITIONAL_SIZES = [117, 126, 135, 144]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(4, 6, 8), required=True)
    parser.add_argument("--capture-max", type=int, choices=(112, 144), required=True)
    args = parser.parse_args()

    source = args.capsule.resolve(strict=True)
    output = args.output.resolve()
    if not source.is_dir() or not (source / "gamma4-configs.json").is_file():
        parser.error("--capsule must contain the teacher's gamma4-configs.json")
    if output == source or source in output.parents or output.exists():
        parser.error("--output must be a new directory outside the source capsule")

    source_config = (source / "gamma4-configs.json").read_bytes()
    config = json.loads(source_config)
    engine = config["q35-mtp4"]
    sizes = engine["compilation_config"]["cudagraph_capture_sizes"]
    if sizes != ORIGINAL_CAPTURE_SIZES:
        parser.error("source capsule does not have the frozen 48-entry capture list")
    if engine["speculative_config"]["num_speculative_tokens"] != 4:
        parser.error("source capsule does not have native MTP4 as its base")

    shutil.copytree(
        source,
        output,
        ignore=shutil.ignore_patterns("evidence", "cache", "venv", "__pycache__"),
    )
    engine["speculative_config"]["num_speculative_tokens"] = args.depth
    if args.capture_max == 144:
        engine["compilation_config"]["cudagraph_capture_sizes"] = (
            ORIGINAL_CAPTURE_SIZES + MTP8_ADDITIONAL_SIZES
        )
    (output / "gamma4-configs.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "native-depth-profile.json").write_text(
        json.dumps(
            {
                "source_config_sha256": hashlib.sha256(source_config).hexdigest(),
                "native_num_speculative_tokens": args.depth,
                "capture_size_count": len(engine["compilation_config"]["cudagraph_capture_sizes"]),
                "max_cudagraph_capture_size": args.capture_max,
                "added_capture_sizes": MTP8_ADDITIONAL_SIZES if args.capture_max == 144 else [],
            },
            indent=2,
        )
        + "\n"
    )
    (output / "evidence").mkdir(exist_ok=True)
    print(output)


if __name__ == "__main__":
    main()
