"""Small call adapters for host APIs that changed across tested revisions."""

from __future__ import annotations

import inspect
from typing import Any


def call_with_supported_kwargs(
    callable_object: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    parameters = inspect.signature(callable_object).parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return callable_object(*args, **kwargs)
    supported_names = {parameter.name for parameter in parameters}
    return callable_object(
        *args,
        **{name: value for name, value in kwargs.items() if name in supported_names},
    )


__all__ = ["call_with_supported_kwargs"]
