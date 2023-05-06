from __future__ import annotations

import contextlib
import contextvars


_active_scope = contextvars.ContextVar("scope")


class _Scope:
    def domain():
        pass


@contextlib.contextmanager
def scope_context():
    scope = _Scope()
    token = _active_scope.set(scope)
    try:
        yield scope
    finally:
        _active_scope.reset(token)


def cross():
    pass


def project(t, mask):
    return tuple(t[i] for i, t in enumerate(t) if mask[i])
