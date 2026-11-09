"""Launch a command with the selected installed backend's custom operators."""
import argparse
import os
from pathlib import Path

from .backends.experimental_paths import configure_custom_opp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-package', type=Path, required=True,
                        help='Site-packages directory containing the selected vllm_ascend')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    env = dict(os.environ)
    configure_custom_opp(env, args.backend_package)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
