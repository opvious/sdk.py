from __future__ import annotations

import contextvars
import dataclasses
import itertools
from typing import Any, Generator, Tuple, TypeVar


_V = TypeVar("_V")


Lazy = Generator[_V, None, None]


def _run_lazy(lazy: Lazy[_V]) -> _V:
    elems = list(itertools.islice(lazy, 2))
    if not elems:
        raise ValueError("Empty iterator")
    if len(elems) > 1:
        raise ValueError("Iterator contained multiple values")
    return elems[0]


@dataclasses.dataclass
class _Scope:
    declarations: list[Any]


_active_scope: Any = contextvars.ContextVar("lazy_scope")


def force(lazy: Lazy[_V]) -> Tuple[_V, list[Any]]:
    scope = _Scope([])
    token = _active_scope.set(scope)
    try:
        value = _run_lazy(lazy)
    finally:
        _active_scope.reset(token)
    return value, scope.declarations


def declare(declaration: Any) -> Any:
    scope = _active_scope.get()
    scope.declarations.append(declaration)
    return declaration
