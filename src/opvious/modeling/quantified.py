from __future__ import annotations

from collections.abc import Iterator
import contextvars
import dataclasses
import itertools
from typing import Any


type Quantified[V] = Iterator[V]


def _run_quantified[V](quantified: Quantified[V]) -> V:
    elems = list(itertools.islice(quantified, 2))
    if not elems:
        raise ValueError("Empty quantified")
    if len(elems) > 1:
        raise ValueError("Quantified contained multiple values")
    return elems[0]


@dataclasses.dataclass
class _Scope:
    declarations: list[Any]


_active_scope: Any = contextvars.ContextVar("quantified_scope")


def unquantify[V](quantified: Quantified[V]) -> tuple[V, list[Any]]:
    scope = _Scope([])
    token = _active_scope.set(scope)
    try:
        value = _run_quantified(quantified)
    finally:
        _active_scope.reset(token)
    return value, scope.declarations


def declare(declaration: Any) -> Any:
    scope = _active_scope.get()
    scope.declarations.append(declaration)
    return declaration
