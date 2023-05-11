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
)

from ..common import Label


Name = str


Environment = KeysView[Name]


@dataclasses.dataclass(eq=False, frozen=True)
class Identifier:
    name: Optional[Name]
    """The requested name"""

    def format(self) -> Name:
        """Generates the final name"""
        return _active_scope.get().format(self)


class GlobalIdentifier(Identifier):
    pass


@dataclasses.dataclass(eq=False, frozen=True)
class DimensionIdentifier(GlobalIdentifier):
    label: Optional[Label]


TensorVariant = Literal["variable", "parameter"]


@dataclasses.dataclass(eq=False, frozen=True)
class TensorIdentifier(GlobalIdentifier):
    variant: TensorVariant
    label: Optional[Label]


@dataclasses.dataclass(eq=False, frozen=True)
class AliasIdentifier(GlobalIdentifier):
    pass


class QuantifierIdentifier(Identifier):
    @property
    def quantifiable(self) -> Any:
        raise NotImplementedError()

    @classmethod
    def root(
        cls, quantifiable: Any, name: Optional[Name] = None
    ) -> QuantifierIdentifier:
        return _RootQuantifierIdentifier(name=name, quantifiable=quantifiable)

    def child(self, name: Optional[Name]) -> QuantifierIdentifier:
        if name is None or name == self.name:
            return self
        return _ChildQuantifierIdentifier(name=name, parent=self)


@dataclasses.dataclass(eq=False, frozen=True)
class _RootQuantifierIdentifier(QuantifierIdentifier):
    quantifiable: Any


@dataclasses.dataclass(eq=False, frozen=True)
class _ChildQuantifierIdentifier(QuantifierIdentifier):
    parent: QuantifierIdentifier

    @property
    def quantifiable(self) -> Any:
        return self.parent.quantifiable


class IdentifierFormatter:
    def __init__(self) -> None:
        self.__globals: dict[GlobalIdentifier, bool] = {}

    def formatted_globals(self) -> Mapping[GlobalIdentifier, bool]:
        return self.__globals

    def format(self, identifier: Identifier, env: Environment) -> Name:
        if isinstance(identifier, GlobalIdentifier):
            self.__globals[identifier] = True
        if isinstance(identifier, DimensionIdentifier):
            return self._format_dimension(identifier, env)
        elif isinstance(identifier, TensorIdentifier):
            return self._format_tensor(identifier, env)
        elif isinstance(identifier, AliasIdentifier):
            return self._format_alias(identifier, env)
        elif isinstance(identifier, QuantifierIdentifier):
            return self.format_quantifier(identifier, env)
        else:
            raise TypeError(f"Unexpected identifier: {identifier}")

    def _format_dimension(
        self, dim: DimensionIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()

    def _format_tensor(
        self, tensor: TensorIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()

    def _format_alias(self, alias: AliasIdentifier, env: Environment) -> Name:
        raise NotImplementedError()

    def format_quantifier(
        self, quant: QuantifierIdentifier, env: Environment
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
            while isinstance(identifier, _ChildQuantifierIdentifier):
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
        scope.prepare(quantifier)
    token = _active_scope.set(child)
    try:
        yield
    finally:
        _active_scope.reset(token)
