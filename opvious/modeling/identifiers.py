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
    Mapping,
    Optional,
    Sequence,
)

from ..common import Label


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


@dataclasses.dataclass(eq=False, frozen=True)
class TensorIdentifier(GlobalIdentifier):
    name: Optional[Name]
    is_parameter: bool


@dataclasses.dataclass(eq=False, frozen=True)
class AliasIdentifier(GlobalIdentifier):
    name: Name


@dataclasses.dataclass(eq=False, frozen=True)
class QuantifierGroup:
    alias: AliasIdentifier
    subscripts: tuple[Any, ...]
    rank: int


class QuantifierIdentifier(Identifier):
    space: Any  # ScalarSpace
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
            try:
                label = self._labels[identifier]
            except KeyError as exc:
                raise Exception(
                    "Unknown identifier, did you forget to add a dependency?"
                ) from exc
            if isinstance(identifier, DimensionIdentifier):
                return self._format_dimension(label, env)
            elif isinstance(identifier, TensorIdentifier):
                if identifier.is_parameter:
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


class DefaultIdentifierFormatter(IdentifierFormatter):
    def __init__(self, labels: Mapping[GlobalIdentifier, Label]) -> None:
        super().__init__(labels)

    def _format_dimension(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        if i is None:
            return _first_available(label[0].upper(), env)
        return f"{label[i]}^\\mathrm{{{label[:i]}}}" if i > 0 else label[i]

    def _format_parameter(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        if not i:
            return _first_available(label[0].lower(), env)
        return f"{label[i].lower()}^\\mathrm{{{label[:i]}}}"

    def _format_variable(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        r = label[i or 0].lower()
        g = _greek_letters.get(r, r)
        if not i:
            return _first_available(g, env)
        return f"{g}^\\mathrm{{{label[:i]}}}"

    def format_quantifier(
        self, identifier: QuantifierIdentifier, env: Environment
    ) -> Name:
        name = identifier.name
        if not name:
            sp = identifier.space
            group = None
            for g in identifier.groups:
                if g.rank == 1:
                    # Single rank alias
                    group = g
                    break
            if not group and hasattr(sp, "identifier") and sp.identifier:
                # Dimension
                name = _lower_principal(sp.identifier.format())
            else:
                # Interval, possibly aliased
                if not group:
                    group = identifier.outer_group
                if group:
                    name = _lower_principal(group.alias.format())
        return _first_available(name or _DEFAULT_QUANTIFIER_NAME, env)


_DEFAULT_QUANTIFIER_NAME = "x"


def _first_available(name: Name, env: Environment) -> Name:
    while name in env:
        name += "'"
    return name


def _last_capital_index(label: Label) -> Optional[int]:
    j = None
    for i, c in enumerate(label):
        if c.isupper():
            j = i
    return j


def _lower_principal(name: Name) -> Name:
    if "^" not in name:
        return name.lower()
    parts = name.split("^", 1)
    return f"{parts[0].lower()}^{parts[1]}"


_greek_letters = {
    "a": "\\alpha",
    "b": "\\beta",
    "c": "\\chi",
    "d": "\\delta",
    "e": "\\epsilon",
    "f": "\\phi",
    "g": "\\gamma",
    "h": "\\eta",
    "i": "\\iota",
    "j": "\\xi",  # TODO: Find better alternative
    "k": "\\kappa",
    "l": "\\lambda",
    "m": "\\mu",
    "n": "\\nu",
    "o": "\\omicron",
    "p": "\\pi",
    "q": "\\theta",
    "r": "\\rho",
    "s": "\\sigma",
    "t": "\\tau",
    "u": "\\psi",
    "v": "\\zeta",  # TODO: Find better alternative
    "w": "\\omega",
    "x": "\\xi",
    "y": "\\upsilon",
    "z": "\\zeta",
}
