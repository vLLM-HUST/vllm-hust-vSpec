"""Launch a command with the selected installed backend's custom operators."""

import argparse
import importlib.util
import os
import shlex
import shutil
import sys
from pathlib import Path

from .backends.experimental_paths import configure_custom_opp

QWEN35_MTP_COMPAT_ENV = "QWEN35_MTP_COMPAT"


def _configure_qwen35_mtp_compat(env: dict[str, str]) -> None:
    """Add a standalone startup hook without importing vSpec in the child."""
    compat_dir = Path(__file__).resolve().parent / "_vendor/qwen35_mtp_compat"
    if not all(
        (compat_dir / name).is_file()
        for name in (
            "sitecustomize.py",
            "qwen35_mtp_kv_compat.py",
            "qwen35_mtp_mamba_compat.py",
        )
    ):
        raise FileNotFoundError(f"Qwen3.5 MTP compatibility assets missing: {compat_dir}")
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(compat_dir) + (os.pathsep + previous if previous else "")
    env[QWEN35_MTP_COMPAT_ENV] = "1"


def _same_python(executable: Path) -> bool:
    """A shared base interpreter is not enough: the virtual environment must match."""
    expected = Path(sys.executable).absolute()
    executable = executable.absolute()
    return executable.parent == expected.parent and executable.resolve() == expected.resolve()


def _require_matching_command(command: list[str]) -> None:
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(f"Command not found: {command[0]}")
    path = Path(executable)
    if _same_python(path):
        return
    try:
        with path.open("rb") as stream:
            first_line = stream.readline(512).decode("utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise RuntimeError("Cannot verify the command Python interpreter") from exc
    if not first_line.startswith("#!"):
        raise RuntimeError("Command must use the same Python environment as vSpec")
    parts = shlex.split(first_line[2:].strip())
    if not parts:
        raise RuntimeError("Command has an empty interpreter directive")
    interpreter = parts[0]
    if Path(interpreter).name == "env":
        if len(parts) != 2 or parts[1].startswith("-"):
            raise RuntimeError("Cannot verify the command Python interpreter")
        interpreter = shutil.which(parts[1]) or ""
    if not interpreter or not _same_python(Path(interpreter)):
        raise RuntimeError("Command uses a different Python environment than vSpec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-package",
        type=Path,
        required=True,
        help="Site-packages directory containing the selected vllm_ascend",
    )
    parser.add_argument(
        "--qwen35-mtp-compat",
        action="store_true",
        help="Enable the packaged, narrow Qwen3.5 MTP KV/Mamba startup bridge",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    spec = importlib.util.find_spec("vllm_ascend")
    backend_package = args.backend_package.resolve()
    if (
        spec is None
        or spec.origin is None
        or not Path(spec.origin).resolve().is_relative_to(backend_package / "vllm_ascend")
    ):
        raise RuntimeError("Active Python environment does not import the selected vllm_ascend")
    _require_matching_command(command)
    env = dict(os.environ)
    configure_custom_opp(env, backend_package)
    if args.qwen35_mtp_compat:
        _configure_qwen35_mtp_compat(env)
    else:
        env.pop(QWEN35_MTP_COMPAT_ENV, None)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
