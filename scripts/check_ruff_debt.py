"""Keep existing Ruff debt fixed while requiring clean new or edited Python files.

The baseline records only files that currently fail lint or formatting. A changed
file must be brought into compliance; new diagnostics cannot be silently added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "ruff-debt-baseline.json"


def ruff_findings(*args: str) -> set[str]:
    result = subprocess.run(
        ["ruff", *args, "--output-format", "json", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr or result.stdout)
    return {
        str(Path(item["filename"]).resolve().relative_to(ROOT))
        for item in json.loads(result.stdout)
    }


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def current_debt() -> dict[str, dict[str, str]]:
    return {
        "lint": {path: digest(path) for path in sorted(ruff_findings("check"))},
        "format": {path: digest(path) for path in sorted(ruff_findings("format", "--check"))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    version = subprocess.check_output(["ruff", "--version"], text=True).strip()
    debt = current_debt()
    if args.write_baseline:
        BASELINE.write_text(json.dumps({"ruff": version, **debt}, indent=2) + "\n")
        return
    baseline = json.loads(BASELINE.read_text())
    if baseline["ruff"] != version:
        raise SystemExit(f"Ruff version changed: expected {baseline['ruff']}, got {version}")
    failures = []
    for kind, files in debt.items():
        for path, sha256 in files.items():
            if baseline[kind].get(path) != sha256:
                failures.append(f"{kind}: {path}")
    if failures:
        raise SystemExit("New or changed Ruff debt:\n" + "\n".join(failures))
    print(
        "Ruff regression check passed "
        f"({len(debt['lint'])} unchanged lint-debt files, "
        f"{len(debt['format'])} unchanged format-debt files)."
    )


if __name__ == "__main__":
    main()
