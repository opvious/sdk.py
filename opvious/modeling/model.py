from __future__ import annotations

import dataclasses
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from ..common import Label, to_camel_case
from .ast import (
    Expression,
    ExpressionLike,
    ExpressionReference,
    literal,
    Predicate,
    ScalarQuantifiable,
    Quantifiable,
    QuantifiableReference,
    Quantification,
    Quantifier,
    cross,
    domain_from_quantifiable,
    expression_quantifiable,
    is_literal,
    render_identifier,
    to_expression,
    within_domain,
)
from .identifiers import (
    AliasIdentifier,
    DimensionIdentifier,
    Environment,
    GlobalIdentifier,
    IdentifierFormatter,
    Name,
    QuantifierIdentifier,
    TensorIdentifier,
    TensorVariant,
    local_formatting_scope,
    global_formatting_scope,
)
from .images import Image
from .quantified import Quantified


class _Definition:
    @property
    def label(self) -> Optional[Label]:
        raise NotImplementedError()

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        raise NotImplementedError()

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _Statement:
    label: Label
    definition: _Definition
    inherited: bool


class Model:
    """Base model class"""

    def _gather_statements(self) -> Sequence[_Statement]:
        statements: list[_Statement] = []

        def visit(dct, inherited):
            for attr, value in dct.items():
                if isinstance(value, property):
                    value = value.fget
                if not isinstance(value, _Definition):
                    continue
                label = value.label or to_camel_case(attr)
                statements.append(_Statement(label, value, inherited))

        visit(self.__dict__, False)
        for i, cls in enumerate(self.__class__.__mro__):
            visit(cls.__dict__, i > 0)
        return statements

    def render_specification_source(
        self,
        formatter_factory: Optional[
            Callable[[Mapping[GlobalIdentifier, Label]], IdentifierFormatter]
        ] = None,
        labels: Optional[Iterable[Label]] = None,  # TODO: Implement
        include_inherited=False,  # TODO: Implement
    ) -> str:
        statements = self._gather_statements()
        allowlist = set(labels or [])
        by_identifier = {
            s.definition.identifier: s
            for s in statements
            if s.definition.identifier
        }
        labels_by_identifier = {
            i: d.label for i, d in by_identifier.items() if d.label
        }
        if formatter_factory:
            formatter = formatter_factory(labels_by_identifier)
        else:
            formatter = _DefaultFormatter(labels_by_identifier)
        reserved = {i.name: i for i in labels_by_identifier if i.name}
        with global_formatting_scope(formatter, reserved):
            idens = set()
            rendered: list[Optional[str]] = []
            for s in statements:
                if not s.label or (allowlist and s.label not in allowlist):
                    continue
                rs = s.definition.render_statement(s.label, self)
                if not rs:
                    continue
                rendered.append(rs)
                if s.definition.identifier:
                    idens.add(s.definition.identifier)
            for iden in formatter.formatted_globals():
                if iden in idens:
                    continue
                s = by_identifier[iden]
                rs = s.definition.render_statement(s.label, self)
                if not rs:
                    raise Exception("Missing rendered statement")
                rendered.append(rs)
            contents = "".join(f"{s} \\\\\n" for s in rendered if s)
        return f"$$\n\\begin{{align}}\n{contents}\\end{{align}}\n$$"

    def _repr_latex_(self) -> str:
        return self.render_specification_source()


class _DefaultFormatter(IdentifierFormatter):
    def __init__(self, labels: Mapping[GlobalIdentifier, Label]) -> None:
        super().__init__(labels)

    def _format_dimension(self, label: Label, env: Environment) -> Name:
        return f"D^{{{label}}}"

    def _format_parameter(self, label: Label, env: Environment) -> Name:
        return f"p^{{{label}}}"

    def _format_variable(self, label: Label, env: Environment) -> Name:
        return f"v^{{{label}}}"

    def format_quantifier(
        self, identifier: QuantifierIdentifier, env: Environment
    ) -> Name:
        name = identifier.name
        if not name:
            quantifiable = identifier.quantifiable
            if isinstance(quantifiable, QuantifiableReference):
                name = quantifiable.identifier.format().lower()
        return self._first_available(name or "i", env)

    def _first_available(self, name: Name, env: Environment) -> Name:
        while name in env:
            name += "x"
        return name


class Dimension(_Definition, ScalarQuantifiable):
    """An abstract collection of values

    Args:
        label: Dimension label override. By default the label is derived from
            the attribute's name
        name: The dimension's name. By default the name will be derived from
            the dimension's label
        is_numeric: Whether the dimension will only contain integers. This
            enables arithmetic operations on this dimension's quantifiers

    As a convenience, iterating on a dimension returns a suitable quantifier.
    This allows creating simple constraints directly:

    .. code-block:: python

        class MyModel(Model):
            products = Dimension()
            product_count = Variable(products, image=natural())

            @constraint()
            def at_least_one_of_each_product(self):
                for p in self.products:  # <=
                    yield self.product_count(p) >= 1

    Dimensions must be set as attributes on :class:`Model` instances so that
    they can be automatically picked up.
    """

    def __init__(
        self,
        label: Optional[Label] = None,
        name: Optional[Name] = None,
        is_numeric: bool = False,
    ):
        self._identifier = DimensionIdentifier(name=name)
        self._label = label
        self._is_numeric = is_numeric

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def render(self) -> str:
        return self._identifier.format()

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        s = f"\\S^d_{{{label}}}&: {self._identifier.format()}"
        if self._is_numeric:
            s += " \\subset \\mathbb{Z}"
        return s


@dataclasses.dataclass(frozen=True)
class _Interval(ScalarQuantifiable):
    lower_bound: Expression
    upper_bound: Expression

    def render(self) -> str:
        lb = self.lower_bound
        ub = self.upper_bound
        if is_literal(lb, 0) and is_literal(ub, 1):
            return "\\{0, 1\\}"
        return f"\\{{ {lb.render()} \\ldots {ub.render()} \\}}"


_integers = _Interval(literal(-math.inf), literal(math.inf))


def interval(
    lower_bound: ExpressionLike, upper_bound: ExpressionLike
) -> Quantified[Quantifier]:
    """A range of values

    Args:
        lower_bound: The range's inclusive lower bound
        upper_bound: The range's inclusive upper bound
    """
    interval = _Interval(
        lower_bound=to_expression(lower_bound),
        upper_bound=to_expression(upper_bound),
    )
    return iter(interval)


class _Tensor(_Definition):
    def __init__(
        self,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        image: Image = Image(),
    ):
        self._identifier = TensorIdentifier(name=name, variant=self._variant)
        self._domain = domain_from_quantifiable(quantifiables)
        self._image = image
        self._label = label

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def quantification(self) -> Quantification:
        return cross(self._domain)

    @property
    def _variant(self) -> TensorVariant:
        raise NotImplementedError()

    def __call__(self, *subscripts: Expression) -> Expression:
        return ExpressionReference(self._identifier, subscripts)

    def to_expression(self) -> Expression:
        return self()

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        c = self._variant[0]
        s = f"\\S^{c}_{{{label}}}&: {self._identifier.format()} \\in "
        s += self._image.render()
        domain = self._domain
        if domain.quantifiers:
            with local_formatting_scope(domain.quantifiers):
                if domain.mask:
                    sup = domain.render()
                else:
                    formatted = [
                        q.quantifiable.render() for q in domain.quantifiers
                    ]
                    sup = " \\times ".join(formatted)
                s += f"^{{{sup}}}"
        return s


class Parameter(_Tensor):
    """An optimization input parameter"""

    _variant = "parameter"


class Variable(_Tensor):
    """An optimization output variable"""

    _variant = "variable"


_M = TypeVar("_M", bound=Model, contravariant=True)


_R = TypeVar(
    "_R",
    bound=Union[Expression, Quantification, Quantified[Quantifier]],
    covariant=True,
)


class Aliasable(Protocol[_M, _R]):
    def __call__(self, model: _M, *expressions: Expression) -> _R:
        pass


_AliasedVariant = Literal[
    "expression",
    "scalar_quantification",
    "quantification",
]


@dataclasses.dataclass(frozen=True)
class _Aliased:
    variant: _AliasedVariant
    quantifiables: Sequence[Optional[ScalarQuantifiable]]


class Alias(_Definition):
    label = None

    def __init__(
        self,
        aliasable: Aliasable[Any, Any],
        name: Name,
        subscript_names: Optional[Iterable[Name]] = None,
    ):
        self._identifier = AliasIdentifier(name=name)
        self._subscript_names = subscript_names
        self._aliasable = aliasable
        self._aliased: Optional[_Aliased] = None

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    def render_statement(self, _label: Label, model: Any) -> Optional[str]:
        if self._aliased is None:
            return None  # Not used
        quantifiable = tuple(
            q or _integers for q in self._aliased.quantifiables
        )
        outer_domain = domain_from_quantifiable(
            quantifiable, names=self._subscript_names
        )
        expressions = [Quantifier(q) for q in outer_domain.quantifiers]
        value = self._aliasable(model, *expressions)
        s = "\\S^a&: "
        with local_formatting_scope(outer_domain.quantifiers):
            if outer_domain.quantifiers:
                s += f"\\forall {outer_domain.render()}, "
            s += render_identifier(self._identifier, *expressions)
            s += " \\doteq "
            if self._aliased.variant == "expression":
                s += value.render()
            else:
                inner_domain = domain_from_quantifiable(value)
                with local_formatting_scope(inner_domain.quantifiers):
                    if len(inner_domain.quantifiers) > 1 or inner_domain.mask:
                        s += f"\\{{ {inner_domain.render()} \\}}"
                    else:
                        s += inner_domain.quantifiers[0].quantifiable.render()
        return s

    def __call__(self, model: Any, *expressions: Expression) -> Any:
        # This is used for property calls
        if self._aliased is None:
            value = self._aliasable(model, *expressions)
            if isinstance(value, Expression):
                variant: _AliasedVariant = "expression"
            elif isinstance(value, tuple):
                variant = "quantification"
            else:
                variant = "scalar_quantification"
            self._aliased = _Aliased(
                variant,
                [expression_quantifiable(x) for x in expressions],
            )
        if self._aliased.variant == "expression":
            return ExpressionReference(self._identifier, expressions)
        else:
            ref = QuantifiableReference(self._identifier, expressions)
            if self._aliased.variant == "quantification":
                return cross(ref)
            else:
                return iter(ref)

    def __get__(self, model: Any, _objtype=None) -> Callable[..., Any]:
        # This is needed for non-property calls
        if not isinstance(model, Model):
            raise TypeError(f"Unexpected model: {model}")

        def wrapped(*expressions: Expression) -> Any:
            return self(model, *expressions)

        return wrapped


_F = TypeVar("_F", bound=Callable[..., Union[Expression, Quantifiable]])


def alias(
    name: Name, subscript_names: Optional[Iterable[Name]] = None
) -> Callable[[_F], _F]:  # TODO: Tighten argument type
    """Decorator creating an alias for the given method

    Args:
        name: The generated alias' name
        subscript_names: Optional names to use for the alias' quantifiers


    The decorated function may be wrapped as a property.
    """

    def wrap(fn):
        return Alias(fn, name=name, subscript_names=subscript_names)

    return wrap


ConstraintBody = Callable[[_M], Quantified[Predicate]]


class Constraint(_Definition):
    """Optimization constraint

    Constraint should be created via the :func:`.constraint` decorator.
    """

    identifier = None

    def __init__(self, body: ConstraintBody, label: Optional[Label] = None):
        self._body = body
        self._label = label

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

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


@overload
def constraint(body: ConstraintBody) -> Constraint:
    ...


@overload
def constraint(
    *,
    label: Optional[Label] = None,
) -> Callable[[ConstraintBody], Constraint]:
    ...


def constraint(
    body: Optional[ConstraintBody] = None, *, label: Optional[Label] = None
) -> Any:
    """Decorator flagging a model method as a constraint

    Args:
        label: Constraint label override. By default the label is derived from
            the method's name.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class MyModel(Model):
            # ...

            @constraint
            def ensure_something(self):
                # ...

            @constraint(label="ensuresEverything")
            def ensure_something_else(self):
                # ...
    """
    if body:
        return Constraint(body)

    def wrap(fn):
        return Constraint(fn, label=label)

    return wrap


ObjectiveSense = Literal["max", "min"]
"""Optimization direction"""


ObjectiveBody = Callable[[_M], Expression]


class Objective(_Definition):
    """Optimization objective

    Objectives should be created via the :func:`.objective` decorator.
    """

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

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        sense = self._sense
        if sense is None:
            if label.startswith("min"):
                sense = "min"
            elif label.startswith("max"):
                sense = "max"
            else:
                raise Exception(f"Missing sense for objective {label}")
        expression = to_expression(self._body(model))
        return f"\\S^o_{{{label}}}&: \\{sense} {expression.render()}"


@overload
def objective(body: ObjectiveBody) -> Constraint:
    ...


@overload
def objective(
    *, sense: Optional[ObjectiveSense] = None, label: Optional[Label] = None
) -> Callable[[ObjectiveBody], Constraint]:
    ...


def objective(
    body: Optional[ObjectiveBody] = None,
    *,
    sense: Optional[ObjectiveSense] = None,
    label: Optional[Label] = None,
) -> Any:
    """Decorator flagging a method as an objective

    Args:
        sense: Optimization direction. This may be omitted if the method name
            starts with `minimize` or `maximize`, in which case the appropriate
            sense will be inferred.
        label: Objective label override. By default the label is derived from
            the method's name.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class MyModel(Model):
            # ...

            @objective
            def minimize_this(self):
                # ...

            @objective(sense="max")
            def optimize_that(self):
                # ...
    """

    def wrap(fn):
        return Objective(fn, sense=sense, label=label)

    return wrap
