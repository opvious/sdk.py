from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from ..common import Label, to_camel_case
from .ast import cross, Domain, Expression, locally, Predicate
from .identifier import HasIdentifier, Identifier
from .image import Image


Name = str


@dataclasses.dataclass(frozen=True)
class _GlobalIdentifier(Identifier):
    label: Label
    name: Optional[Name] = None


@dataclasses.dataclass(frozen=True)
class _Definition(HasIdentifier):
    model: Model
    identifier: _GlobalIdentifier

    @property
    def label(self):
        return self.identifier.label

    @property
    def category(self) -> str:
        raise NotImplementedError()

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class DimensionDefinition(_Definition):
    category = "d"

    def __iter__(self):
        return (t[0] for t in cross(self))

    def render(self) -> str:
        return self.identifier.render()


@dataclasses.dataclass(frozen=True)
class Dimension:
    name: Optional[Name] = None
    label: Optional[Label] = None


@dataclasses.dataclass(frozen=True)
class _TensorDefinition(_Definition):
    image: Image
    sources: Tuple[HasIdentifier, ...]

    def render(self) -> str:
        return rf"{self.identifier.render()} \in \mathbb{{R}}"


@dataclasses.dataclass(frozen=True)
class ParameterDefinition(_TensorDefinition):
    category = "p"


@dataclasses.dataclass(frozen=True)
class VariableDefinition(_TensorDefinition):
    category = "v"


class _Tensor:
    sources: Tuple[HasIdentifier, ...]
    image: Optional[Image]
    label: Optional[Label]
    name: Optional[Name]

    def __init__(self, *sources, image=None, name=None, label=None):
        self.sources = sources
        self.image = image
        self.name = name
        self.label = label


@dataclasses.dataclass(frozen=True)
class Parameter(_Tensor):
    pass


@dataclasses.dataclass(frozen=True)
class Variable(_Tensor):
    pass


@dataclasses.dataclass(frozen=True)
class ConstraintDefinition(_Definition):
    category = "c"

    domain: Domain
    predicate: Predicate

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class Constraint:
    predicate: Predicate
    domain: Domain
    label: Optional[Label] = None


ObjectiveSense = Literal["max", "min"]


@dataclasses.dataclass(frozen=True)
class ObjectiveDefinition(_Definition):
    category = "o"
    sense: ObjectiveSense
    expression: Expression

    def render(self) -> str:
        return rf"\\{self.sense} {self.expression.render()}"


@dataclasses.dataclass(frozen=True)
class Objective:
    sense: ObjectiveSense
    expression: Expression
    label: Optional[Label] = None


DefinitionValue = Union[
    Dimension,
    Parameter,
    Variable,
    Constraint,
    Objective,
]


class Model:
    if TYPE_CHECKING:

        def __getattr__(self, attr: str) -> Any:
            return self.__dict__[attr]

    def __setattr__(self, attr: str, value: DefinitionValue) -> None:
        if getattr(self, attr):
            raise Exception(f"Duplicate attribute: {attr}")
        definition: _Definition
        if isinstance(value, Dimension):
            definition = DimensionDefinition(
                model=self,
                identifier=_GlobalIdentifier(
                    label=value.label or to_camel_case(attr),
                    name=value.name,
                ),
            )
        elif isinstance(value, (Parameter, Variable)):
            identifier = _GlobalIdentifier(
                label=value.label or to_camel_case(attr),
                name=value.name,
            )
            Definition = (
                ParameterDefinition
                if isinstance(value, Parameter)
                else VariableDefinition
            )
            definition = Definition(
                model=self,
                sources=value.sources,
                image=value.image or Image(),
                identifier=identifier,
            )
        elif isinstance(value, Constraint):
            definition = ConstraintDefinition(
                model=self,
                predicate=value.predicate,
                domain=value.domain,
                identifier=_GlobalIdentifier(
                    label=value.label or to_camel_case(attr),
                ),
            )
        elif isinstance(value, Objective):
            definition = ObjectiveDefinition(
                model=self,
                sense=value.sense,
                expression=value.expression,
                identifier=_GlobalIdentifier(
                    label=value.label or to_camel_case(attr),
                ),
            )
        else:
            raise TypeError("Unexpected definition value")
        self.__dict__[attr] = definition

    def render(self):
        return _render_definitions(self.__dict__.values())


def define(model: Model, attr: str, value: DefinitionValue) -> Any:
    setattr(model, attr, value)
    return getattr(model, attr)


def constrain(model, label=None):
    def wrap(fn):
        predicate, domain = locally(fn())
        constraint = Constraint(
            label=label, predicate=predicate, domain=domain
        )
        define(model, fn.__name__, constraint)
        return fn

    return wrap


def maximize(expression, label=None) -> Objective:
    return Objective(sense="max", label=label, expression=expression)


def minimize(expression, label=None) -> Objective:
    return Objective(sense="min", label=label, expression=expression)


def _render_definition_section(d: _Definition) -> str:
    section = f"\\S^{d.category}"
    if d.label:
        section += f"_{{{d.label}}}"
    return section


def _render_definitions(definitions: list[_Definition]) -> str:
    lines = [
        f"  {_render_definition_section(d)}&: {d.render()} \\\\\n"
        for d in definitions
    ]
    return f"\\begin{{align}}\n{''.join(lines)}end{{align}}"
