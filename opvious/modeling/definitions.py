from __future__ import annotations

import dataclasses
import functools
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from ..common import Label, to_camel_case
from .ast import cross, Domain, Expression, locally, Predicate, reference
from .identifier import (
    GlobalIdentifier,
    LabeledIdentifier,
    global_identifiers,
    HasIdentifier,
    local_identifiers,
    Name,
    Renderer,
    SimpleRenderer,
)
from .image import Image


@dataclasses.dataclass(frozen=True)
class _Definition:
    model: Model
    identifier: GlobalIdentifier

    @property
    def label(self) -> Optional[Label]:
        if isinstance(self.identifier, LabeledIdentifier):
            return self.identifier.label
        return None

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

    def __call__(self, *expressions: Tuple[Expression, ...]) -> Expression:
        return reference(self.identifier, expressions)

    def render(self) -> str:
        return rf"{self.identifier.render()} \in \mathbb{{R}}"


@dataclasses.dataclass(frozen=True)
class ParameterDefinition(_TensorDefinition):
    category = "p"


@dataclasses.dataclass(frozen=True)
class VariableDefinition(_TensorDefinition):
    category = "v"


@dataclasses.dataclass(frozen=True)
class _Tensor:
    sources: Tuple[HasIdentifier, ...]
    image: Optional[Image]
    label: Optional[Label]
    name: Optional[Name]


@dataclasses.dataclass(frozen=True, init=False)
class Parameter(_Tensor):
    def __init__(
        self,
        *sources: Tuple[HasIdentifier, ...],
        image: Optional[Image] = None,
        label: Optional[Label] = None,
        name: Optional[Name] = None
    ) -> None:
        super().__init__(sources, image=image, label=label, name=name)


@dataclasses.dataclass(frozen=True, init=False)
class Variable(_Tensor):
    def __init__(
        self,
        *sources: Tuple[HasIdentifier, ...],
        image: Optional[Image] = None,
        label: Optional[Label] = None,
        name: Optional[Name] = None
    ) -> None:
        super().__init__(sources, image=image, label=label, name=name)


@dataclasses.dataclass(frozen=True)
class ConstraintDefinition(_Definition):
    category = "c"

    domain: Domain
    predicate: Predicate

    def render(self) -> str:
        declarations = self.domain.declarations
        with local_identifiers(declarations):
            if declarations:
                rendered = f"\\forall {self.domain.render()}, "
            else:
                rendered = ""
            rendered += self.predicate.render()
        return rendered


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
        return f"\\{self.sense} {self.expression.render()}"


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
        if hasattr(self, attr):
            raise Exception(f"Duplicate attribute: {attr}")
        definition: _Definition
        if isinstance(value, Dimension):
            definition = DimensionDefinition(
                model=self,
                identifier=LabeledIdentifier(
                    label=value.label or to_camel_case(attr),
                    name=value.name,
                ),
            )
        elif isinstance(value, _Tensor):
            identifier = LabeledIdentifier(
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
                identifier=LabeledIdentifier(
                    label=value.label or to_camel_case(attr),
                ),
            )
        elif isinstance(value, Objective):
            definition = ObjectiveDefinition(
                model=self,
                sense=value.sense,
                expression=value.expression,
                identifier=LabeledIdentifier(
                    label=value.label or to_camel_case(attr),
                ),
            )
        else:
            raise TypeError("Unexpected definition value")
        self.__dict__[attr] = definition


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


def render(model: Model, renderer: Optional[Renderer] = None):
    definitions = list(model.__dict__.values())
    identifiers = list(d.identifier for d in definitions)
    with global_identifiers(renderer or SimpleRenderer(), identifiers):
        lines = [
            f"  {_render_definition_section(d)}&: {d.render()} \\\\\n"
            for d in definitions
        ]
    return f"\\begin{{align}}\n{''.join(lines)}end{{align}}"


def _render_definition_section(d: _Definition) -> str:
    section = f"\\S^{d.category}"
    if d.label:
        section += f"_{{{d.label}}}"
    return section
