from __future__ import annotations

import dataclasses
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

from ..common import Label
from .ast import (
    cross,
    Domain,
    domain_from_space,
    Expression,
    Predicate,
    Quantifiable,
    ExpressionReference,
    SpaceReference,
    Source,
    Space,
    within_domain,
)
from .identifiers import (
    DimensionIdentifier,
    GlobalIdentifier,
    TensorIdentifier,
    TensorVariant,
    AliasIdentifier,
    Name,
    local_formatting_scope,
)
from .images import Image
from .lazy import Lazy


class Definition:
    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        raise NotImplementedError()

    @property
    def label(self) -> Optional[Label]:
        raise NotImplementedError()

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        raise NotImplementedError()


class Dimension(Definition, Quantifiable):
    def __init__(
        self,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        is_numeric: bool = False,
    ):
        self._identifier = DimensionIdentifier(name=name, label=label)
        self._is_numeric = is_numeric

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._identifier.label

    def render(self) -> str:
        return self._identifier.format()

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        s = f"\\S^d_{{{label}}}&: {self._identifier.format()}"
        if self._is_numeric:
            s += " \\subset \\mathbb{Z}"
        return s


@dataclasses.dataclass(frozen=True)
class Interval(Quantifiable):
    lower_bound: int
    upper_bound: int

    def render(self) -> str:
        lb = self.lower_bound
        ub = self.upper_bound
        if lb == 0 and ub == 1:
            return "{0, 1}"
        return f"{{{lb} \\ldots {ub}}}"


class _Tensor(Definition):
    def __init__(
        self,
        *sources: Source,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        image: Image = Image(),
    ):
        self._identifier = TensorIdentifier(
            label=label,
            name=name,
            variant=self._variant,
        )
        self._domain = domain_from_space(cross(sources))
        self._image = image

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._identifier.label

    @property
    def _variant(self) -> TensorVariant:
        raise NotImplementedError()

    def __call__(self, *subscripts: Expression) -> Expression:
        return ExpressionReference(self._identifier, subscripts)

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        c = self._variant[0]
        s = f"\\S^{c}_{{{label}}}&: {self._identifier.format()} \\in "
        s += self._image.render()
        domain = self._domain
        if domain.quantifiers:
            if domain.mask:
                sup = domain.render()
            else:
                sup = " \\times ".join(q.format() for q in domain.quantifiers)
            s += f"^{{{sup}}}"
        return s


class Parameter(_Tensor):
    _variant = "parameter"


class Variable(_Tensor):
    _variant = "variable"


_R = TypeVar("_R", bound=Union[Expression, Space], covariant=True)


class Aliasable(Protocol, Generic[_R]):
    def __call__(self, *expressions: Expression) -> _R:
        pass


class Alias(Definition):
    label = None

    def __init__(
        self,
        aliasable: Aliasable,
        sources: tuple[Source, ...],
        name: Optional[Name] = None,
    ):
        self._sources = sources
        self._identifier = AliasIdentifier(name=name)
        self._aliasable = aliasable
        self._aliased: Union[Expression, Domain, None] = None

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    def render_statement(self, _label: Label, _model: Any) -> Optional[str]:
        if self._aliased is None:
            return None  # Not used
        s = f"\\S^a&: {self._identifier.format()} \\doteq "
        if isinstance(self._aliased, Expression):
            s += self._aliased.render()
        else:
            s += f"{{{self._aliased.render()}}}"
        return s

    def __call__(self, model, *expressions):
        if self.aliased is None:
            self.aliased = self.aliasable(model, *expressions)
        if isinstance(self.aliased, Expression):
            return ExpressionReference(self._identifier, expressions)
        else:
            return SpaceReference(self._identifier, expressions)


def alias(
    *sources: Source, name: Optional[Name] = None
) -> Callable[[Aliasable[_R]], Aliasable[_R]]:
    def wrap(fn):
        return Alias(fn, sources=sources, name=name)

    return wrap


_B = TypeVar("_B")


@dataclasses.dataclass(frozen=True)
class _Forwardable(Generic[_B]):
    body: _B

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)


ConstraintBody = Callable[[Any], Lazy[Predicate]]


@dataclasses.dataclass(frozen=True)
class Constraint(Definition, _Forwardable[ConstraintBody]):
    identifier = None

    def __init__(self, body: ConstraintBody, label: Optional[Label] = None):
        self._body = body
        self._label = label

    @property
    def label(self):
        return self._label

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        s = f"\\S^c_{{{label}}}&: "
        predicate, domain = within_domain(self._body(model))
        with local_formatting_scope(domain.quantifiers):
            if domain.quantifiers:
                s += f"\\forall {domain.render()}, "
            s += predicate.render()
        return s


def constraint(label: Optional[Label] = None):
    def wrap(fn):
        return Constraint(fn, label=label)

    return wrap


ObjectiveSense = Literal["maximize", "minimize"]


ObjectiveBody = Callable[[Any], Expression]


@dataclasses.dataclass(frozen=True)
class Objective(Definition, _Forwardable[ObjectiveBody]):
    identifier = None

    def __init__(
        self,
        body: ObjectiveBody,
        sense: Optional[ObjectiveSense] = None,
        label: Optional[Label] = None,
    ):
        self._body = body
        self._sense = sense
        self._label = label

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        sense = self._sense
        if sense is None:
            if label.startswith("minimize"):
                sense = "minimize"
            elif label.startswith("maximize"):
                sense = "maximize"
            else:
                raise Exception(f"Missing sense for objective {label}")
        expression = self._body(model)
        return f"\\S^o_{{{label}}}&: \\{sense} {expression.render()}"


def objective(
    sense: Optional[ObjectiveSense] = None, label: Optional[Label] = None
):
    def wrap(fn):
        return Objective(fn, sense=sense, label=label)

    return wrap
