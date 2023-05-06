from __future__ import annotations

import dataclasses
from typing import Any, Optional, Protocol, Tuple, Union

from ..common import to_camel_case
from .ast import Domain, Expression, Predicate
from .scope import cross, scope_context


class _Definition(Protocol):
    category: str
    model: Model
    label: str

    def render(self) -> str:
        pass


@dataclasses.dataclass(frozen=True)
class DimensionDefinition(_Definition):
    category = "d"
    model: Model
    label: str
    name: Optional[str]

    def __iter__(self):
        return (t[0] for t in cross(self))

    def render(self) -> str:
        return self.name or f"D^{{{self.label}}}"


@dataclasses.dataclass
class Dimension:
    name: Optional[str] = None
    label: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class TensorDefinition(_Definition):
    category: str
    model: Model
    label: str
    sources: list[DimensionDefinition]
    name: Optional[str]

    def render(self) -> str:
        name = self.name or f"p^{{{self.label}}}"
        return rf"{name} \in \mathbb{{R}}"


class _Tensor:
    model: Model
    label: str
    sources: Tuple[Any]
    name: Optional[str]

    def __init__(self, *sources, name=None, label=None):
        self.sources = sources
        self.name = name
        self.label = label


@dataclasses.dataclass(init=False)
class Parameter(_Tensor):
    category = "p"


@dataclasses.dataclass(init=False)
class Variable(_Tensor):
    category = "v"


@dataclasses.dataclass(frozen=True)
class ConstraintDefinition(_Definition):
    category = "c"
    model: Model
    label: str
    domain: Domain
    predicate: Predicate

    def render(self) -> str:
        name = self.name or f"p^{{{self.label}}}"
        return rf"{name} \in \mathbb{{R}}"


@dataclasses.dataclass(frozen=True)
class Constraint:
    predicate: Predicate
    domain: Domain
    label: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class ObjectiveDefinition(_Definition):
    category = "o"
    model: Model
    label: str
    command: str
    expression: Expression

    def render(self) -> str:
        return rf"{self.command} {self.expression.render()}"


@dataclasses.dataclass(frozen=True)
class Objective:
    is_maximization: bool
    expression: Expression
    label: Optional[str] = None


DefinitionValue = Union[
    Dimension,
    Parameter,
    Variable,
    Constraint,
    Objective,
]


class Model:
    def __setattr__(self, key: str, value: DefinitionValue) -> None:
        if key in self.__dict__:
            raise Exception(f"Duplicate key: {key}")
        if isinstance(value, Dimension):
            definition = DimensionDefinition(
                model=self,
                label=value.label or to_camel_case(key),
                name=value.name,
            )
        elif isinstance(value, (Parameter, Variable)):
            definition = TensorDefinition(
                category=value.category,
                model=self,
                label=value.label or to_camel_case(key),
                name=value.name,
                sources=list(value.sources),
            )
        elif isinstance(value, Constraint):
            definition = ConstraintDefinition(
                model=self,
                label=value.label or to_camel_case(key),
                predicate=value.predicate,
                domain=value.domain,
            )
        elif isinstance(value, Objective):
            definition = ObjectiveDefinition(
                model=self,
                label=value.label or to_camel_case(key),
                command="\\max" if value.is_maximization else "\\min",
                expression=value.expression,
            )
        else:
            raise TypeError("Unexpected definition value")
        self.__dict__[key] = definition

    def render(self):
        return _render_definitions(self.__dict__.values())


def extend(model: Model, label: str, value: DefinitionValue) -> None:
    setattr(model, label, value)
    return getattr(model, label)


def constrain(model, label=None):
    def wrap(fn):
        with scope_context() as scope:
            predicate = fn()
            domain = scope.domain()

        constraint = Constraint(
            label=label, predicate=predicate, domain=domain
        )
        extend(model, fn.__name__, constraint)
        return fn

    return wrap


def maximize(expression, label=None) -> Objective:
    return Objective(is_maximization=True, label=label, expression=expression)


def minimize(expression, label=None) -> Objective:
    return Objective(is_maximization=False, label=label, expression=expression)


def _render_definitions(definitions: list[_Definition]) -> str:
    lines = [
        f"  \\S^{d.category}_{{{d.label}}}&: {d.render()} \\\\\n"
        for d in definitions
    ]
    return f"\\begin{{align}}\n{''.join(lines)}end{{align}}"
