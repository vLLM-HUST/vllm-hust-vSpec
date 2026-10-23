"""Fetch pinned benchmark originals externally without executing archive code."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def checked(raw, sha256, size=None):
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Original checksum mismatch")
    if size is not None and len(raw) != size:
        raise ValueError("Original size mismatch")
    return raw


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name or name == ".":
        raise ValueError("Unsafe original path")
    return path


def destination(path):
    target = path.resolve()
    if target == ROOT or ROOT in target.parents:
        raise ValueError("Use an external destination")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Use an empty destination")
    return target


def fetch(url):
    if not url.startswith("https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/"):
        raise ValueError("Only canonical benchmark URLs are allowed")
    for attempt in range(3):
        try:
            with urlopen(
                Request(url, headers={"User-Agent": "benchmark-custody/1.0"}),
                timeout=60,
            ) as response:
                return response.read()
        except (URLError, TimeoutError) as error:
            print(f"Readback attempt {attempt + 1} failed: {error}", file=sys.stderr)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def collect(config):
    payloads = {}
    for capsule in config["capsules"]:
        url = capsule["manifest_url"]
        raw = checked(fetch(url), capsule["manifest_sha256"])
        manifest = json.loads(raw)
        if manifest["source_repository"] != config["source_repository"]:
            raise ValueError("Source repository mismatch")
        if manifest["source_visibility"] != "public":
            raise ValueError("Non-public source")
        seen = set()
        for row in manifest["files"]:
            name = str(safe_path(row["path"]))
            if name in seen:
                raise ValueError("Duplicate manifest member")
            seen.add(name)
            content = checked(
                fetch(url.rsplit("/", 1)[0] + "/snapshot/" + name),
                row["sha256"],
                row["bytes"],
            )
            if name in payloads and payloads[name] != content:
                raise ValueError("Conflicting original identity")
            payloads[name] = content
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    target = destination(args.destination)
    config = json.loads((ROOT / "BENCHMARK_ORIGINALS.json").read_text())
    payloads = collect(config)
    target.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(content)
    print(f"Verified {len(payloads)} original files; no archive code executed")


if __name__ == "__main__":
    main()
