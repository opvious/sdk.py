from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
from typing import Any, Generator, Optional, Protocol, Sequence, Union

from ..common import Label


Name = str


_active_scope: Any = contextvars.ContextVar("rendering_scope")


class Identifier:
    def render(self) -> str:
        scope = _active_scope.get()
        return scope.environment[self]


@dataclasses.dataclass(frozen=True)
class LabeledIdentifier(Identifier):
    label: Label
    name: Optional[Name] = None


@dataclasses.dataclass(frozen=True)
class AliasIdentifier(Identifier):
    name: Name


GlobalIdentifier = Union[LabeledIdentifier, AliasIdentifier]


@dataclasses.dataclass(frozen=True)
class LocalIdentifier(Identifier):
    source: HasIdentifier


class HasIdentifier(Protocol):
    @property
    def identifier(self) -> Identifier:
        raise NotImplementedError()


class Renderer(Protocol):
    def render_labeled_identifier(self, identifier: LabeledIdentifier) -> str:
        raise NotImplementedError()

    def render_alias_identifier(self, identifier: AliasIdentifier) -> str:
        raise NotImplementedError()

    def render_local_identifier(self, identifier: LocalIdentifier) -> str:
        raise NotImplementedError()


_Environment = collections.ChainMap[Identifier, str]


@dataclasses.dataclass
class _Scope:
    renderer: Renderer
    environment: _Environment


@contextlib.contextmanager
def global_identifiers(
    renderer: Renderer,
    identifiers: Sequence[GlobalIdentifier],
) -> Generator[None, None, None]:
    if _active_scope.get(None):
        raise Exception("Rendering scope already active")
    env: _Environment = collections.ChainMap()
    for iden in identifiers:
        if isinstance(iden, LabeledIdentifier):
            rendered = renderer.render_labeled_identifier(iden)
        else:
            rendered = renderer.render_alias_identifier(iden)
        env[iden] = rendered
    token = _active_scope.set(_Scope(renderer, env))
    try:
        yield
    finally:
        _active_scope.reset(token)


@contextlib.contextmanager
def local_identifiers(
    identifiers: Sequence[LocalIdentifier],
) -> Generator[None, None, None]:
    scope = _active_scope.get()
    env = scope.environment.new_child()
    for iden in identifiers:
        env[iden] = scope.renderer.render_local_identifier(iden)
    token = _active_scope.set(_Scope(scope.renderer, env))
    try:
        yield
    finally:
        _active_scope.reset(token)


# Renderer implementations


class SimpleRenderer(Renderer):
    def render_labeled_identifier(self, identifier: LabeledIdentifier) -> str:
        return identifier.label

    def render_alias_identifier(self, identifier: AliasIdentifier) -> str:
        return identifier.name

    def render_local_identifier(self, identifier: LocalIdentifier) -> str:
        return identifier.source.identifier.render()
