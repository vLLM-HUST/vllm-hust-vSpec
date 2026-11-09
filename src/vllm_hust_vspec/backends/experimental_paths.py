"""Locations of locally provisioned validation artifacts for experimental hooks.

These hooks remain fail-closed: configuring a directory does not bypass any of
their original source, numerical or device validation checks.
"""
import os
from pathlib import Path


def artifact_dir(variable: str) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else Path('/__vspec_artifacts_not_configured__') / variable
