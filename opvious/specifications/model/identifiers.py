from __future__ import annotations

import collections
import contextlib
import contextvars
import dataclasses
from typing import (
    Any,
    Generator,
    Iterable,
    KeysView,
    Literal,
    Mapping,
    Optional,
    Sequence,
)

from ...common import Label


Name = str


Environment = KeysView[Name]


@dataclasses.dataclass(eq=False, frozen=True)
class Identifier:
    def format(self) -> Name:
        """Generates the final name"""
        return _active_scope.get().format(self)


class GlobalIdentifier(Identifier):
    name: Optional[Name]


@dataclasses.dataclass(eq=False, frozen=True)
class DimensionIdentifier(GlobalIdentifier):
    name: Optional[Name]


TensorVariant = Literal["variable", "parameter"]


@dataclasses.dataclass(eq=False, frozen=True)
class TensorIdentifier(GlobalIdentifier):
    name: Optional[Name]
    variant: TensorVariant


@dataclasses.dataclass(eq=False, frozen=True)
class AliasIdentifier(GlobalIdentifier):
    name: Name


@dataclasses.dataclass(eq=False, frozen=True)
class QuantifierGroup:
    alias: AliasIdentifier
    subscripts: tuple[Any, ...]
    rank: int


class QuantifierIdentifier(Identifier):
    space: Any  # Space
    groups: Sequence[QuantifierGroup]
    name: Optional[Name]

    @classmethod
    def base(cls, space: Any) -> QuantifierIdentifier:
        return _BaseQuantifierIdentifier(space=space, groups=[])

    def grouped_within(self, group: QuantifierGroup) -> QuantifierIdentifier:
        return _BaseQuantifierIdentifier(
            space=self.space, groups=[group, *self.groups]
        ).named(self.name)

    def named(self, name: Optional[Name]) -> QuantifierIdentifier:
        if name is None or name == self.name:
            return self
        return _NamedQuantifierIdentifier(name=name, parent=self)

    @property
    def outer_group(self) -> Optional[QuantifierGroup]:
        return self.groups[0] if self.groups else None


@dataclasses.dataclass(eq=False, frozen=True)
class _BaseQuantifierIdentifier(QuantifierIdentifier):
    space: Any
    groups: Sequence[QuantifierGroup]
    name = None


@dataclasses.dataclass(eq=False, frozen=True)
class _NamedQuantifierIdentifier(QuantifierIdentifier):
    name: Name
    parent: QuantifierIdentifier

    @property
    def space(self) -> Any:
        return self.parent.space

    @property
    def groups(self) -> Sequence[QuantifierGroup]:  # type: ignore[override]
        return self.parent.groups


class IdentifierFormatter:
    def __init__(self, labels: Mapping[GlobalIdentifier, Label]) -> None:
        self._formatted: dict[GlobalIdentifier, bool] = {}
        self._labels = labels

    def formatted_globals(self) -> Sequence[GlobalIdentifier]:
        return list(self._formatted)

    def format(self, identifier: Identifier, env: Environment) -> Name:
        if isinstance(identifier, GlobalIdentifier):
            self._formatted[identifier] = True
            if identifier.name:
                return identifier.name
            label = self._labels[identifier]
            if isinstance(identifier, DimensionIdentifier):
                return self._format_dimension(label, env)
            elif isinstance(identifier, TensorIdentifier):
                if identifier.variant == "parameter":
                    return self._format_parameter(label, env)
                else:
                    return self._format_variable(label, env)
        elif isinstance(identifier, QuantifierIdentifier):
            return self.format_quantifier(identifier, env)
        raise TypeError(f"Unexpected identifier: {identifier}")

    def _format_dimension(self, label: Label, env: Environment) -> Name:
        raise NotImplementedError()

    def _format_parameter(self, label: Label, env: Environment) -> Name:
        raise NotImplementedError()

    def _format_variable(self, label: Label, env: Environment) -> Name:
        raise NotImplementedError()

    def format_quantifier(
        self, identifier: QuantifierIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()


@dataclasses.dataclass
class _Scope:
    formatter: IdentifierFormatter
    environment: collections.ChainMap[Name, Identifier]  # To owning identifier
    quantifier_names: collections.ChainMap[QuantifierIdentifier, Name]
    global_names: dict[GlobalIdentifier, Name]

    def child(self) -> _Scope:
        return dataclasses.replace(
            self,
            environment=self.environment.new_child(),
            quantifier_names=self.quantifier_names.new_child(),
        )

    def format(self, identifier: Identifier) -> Name:
        if isinstance(identifier, GlobalIdentifier):
            name = self.global_names.get(identifier)
        else:
            if not isinstance(identifier, QuantifierIdentifier):
                raise TypeError(f"Unexpected identifier: {identifier}")
            name = self.quantifier_names.get(identifier)
            if name is None:
                raise Exception(f"Missing quantifier: {identifier}")
        if name is None:
            name = self.prepare(identifier)
        return name

    def prepare(self, identifier: Identifier) -> Name:
        name = self.formatter.format(identifier, self.environment.keys())
        owner = self.environment.get(name)
        if owner is not None and owner != identifier:
            raise Exception(f"Name collision: {name}")
        if isinstance(identifier, GlobalIdentifier):
            self.global_names[identifier] = name
            self.environment.maps[-1][name] = identifier
        else:
            if not isinstance(identifier, QuantifierIdentifier):
                raise TypeError(f"Unexpected identifier: {identifier}")
            self.environment[name] = identifier
            self.quantifier_names[identifier] = name
            while isinstance(identifier, _NamedQuantifierIdentifier):
                identifier = identifier.parent
                self.quantifier_names[identifier] = name
        return name


_active_scope: Any = contextvars.ContextVar("formatting_scope")


@contextlib.contextmanager
def global_formatting_scope(
    formatter: IdentifierFormatter,
    reserved: Optional[Mapping[Name, GlobalIdentifier]] = None,
) -> Generator[None, None, None]:
    scope = _active_scope.get(None)
    if scope:
        raise Exception("Identifier formatter already active")
    scope = _Scope(
        formatter=formatter,
        environment=collections.ChainMap(dict(reserved) if reserved else {}),
        quantifier_names=collections.ChainMap(),
        global_names={},
    )
    token = _active_scope.set(scope)
    try:
        yield
    finally:
        _active_scope.reset(token)


@contextlib.contextmanager
def local_formatting_scope(
    quantifiers: Iterable[QuantifierIdentifier],
) -> Generator[None, None, None]:
    scope = _active_scope.get(None)
    if not scope:
        raise Exception("Missing active formatter")
    child = scope.child()
    for quantifier in quantifiers:
        child.prepare(quantifier)
    token = _active_scope.set(child)
    try:
        yield
    finally:
        _active_scope.reset(token)
