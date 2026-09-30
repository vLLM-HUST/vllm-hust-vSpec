"""Small call adapters for host APIs that changed across tested revisions."""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any


@cache
def _keyword_support(callable_object: Any) -> tuple[bool, frozenset[str]]:
    parameters = inspect.signature(callable_object).parameters.values()
    accepts_variadic = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    return accepts_variadic, frozenset(parameter.name for parameter in parameters)


def call_with_supported_kwargs(
    callable_object: Any,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    accepts_variadic, supported_names = _keyword_support(callable_object)
    if accepts_variadic:
        return callable_object(*args, **kwargs)
    return callable_object(
        *args,
        **{name: value for name, value in kwargs.items() if name in supported_names},
    )


__all__ = ["call_with_supported_kwargs"]
