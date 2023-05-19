from __future__ import annotations

import dataclasses
import inspect
import itertools
import logging
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from ...common import Label
from .ast import (
    Expression,
    ExpressionLike,
    ExpressionReference,
    literal,
    Predicate,
    Space,
    Quantifiable,
    QuantifiableReference,
    Quantification,
    Quantifier,
    QuantifierIdentifier,
    cross,
    domain_from_quantifiable,
    expression_space,
    is_literal,
    render_identifier,
    to_expression,
    within_domain,
)
from .identifiers import (
    AliasIdentifier,
    DimensionIdentifier,
    GlobalIdentifier,
    Name,
    TensorIdentifier,
    TensorVariant,
    local_formatting_scope,
)
from .images import Image
from .quantified import Quantified, unquantify
from .statements import Definition, Model, ModelFragment


_logger = logging.getLogger(__name__)


class Dimension(Definition, Space):
    """An abstract collection of values

    Args:
        label: Dimension label override. By default the label is derived from
            the attribute's name
        name: The dimension's name. By default the name will be derived from
            the dimension's label
        is_numeric: Whether the dimension will only contain integers. This
            enables arithmetic operations on this dimension's quantifiers

    Dimensions are `Quantifiable` and as such can be quantified over using
    :func:`.cross`. As a convenience, iterating on a dimension also a suitable
    quantifier. This allows creating simple constraints directly:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable(products)

            @constraint
            def at_least_one_of_each(self):
                for p in self.products:  # Note the iteration here
                    yield self.count(p) >= 1
    """

    def __init__(
        self,
        *,
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

    def render_statement(self, label: Label, _owner: Any) -> Optional[str]:
        _logger.debug("Rendering dimension %s...", label)
        s = f"\\S^d_\\mathrm{{{label}}}&: {self._identifier.format()}"
        if self._is_numeric:
            s += " \\subset \\mathbb{Z}"
        return s


@dataclasses.dataclass(frozen=True)
class _Interval(Space):
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


class _Tensor(Definition):
    def __init__(
        self,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        image: Image = Image(),
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        self._identifier = TensorIdentifier(name=name, variant=self._variant)
        self._domain = domain_from_quantifiable(quantifiables)
        self._label = label
        self.image = image
        self.qualifiers = qualifiers

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

    def __call__(self, *subscripts: ExpressionLike) -> Expression:
        return ExpressionReference(
            self._identifier, tuple(to_expression(s) for s in subscripts)
        )

    def render_statement(self, label: Label, _owner: Any) -> Optional[str]:
        _logger.debug("Rendering tensor %s...", label)
        c = self._variant[0]
        s = f"\\S^{c}_\\mathrm{{{_render_label(label, self.qualifiers)}}}&: "
        s += f"{self._identifier.format()} \\in "
        s += self.image.render()
        domain = self._domain
        if domain.quantifiers:
            with local_formatting_scope(domain.quantifiers):
                if domain.mask is None:
                    formatted: list[str] = []
                    for g, qs in itertools.groupby(
                        domain.quantifiers, key=lambda q: q.outer_group
                    ):
                        if g is None:
                            formatted.extend(q.space.render() for q in qs)
                        else:
                            formatted.append(g.alias.format())
                    sup = " \\times ".join(formatted)
                else:
                    sup = f"\\{{ {domain.render()} \\}}"
                s += f"^{{{sup}}}"
        return s


def _render_label(label: Label, qualifiers: Optional[Sequence[Label]]) -> str:
    s = label
    if qualifiers:
        s += f"[{','.join(qualifiers)}]"
    return s


class Parameter(_Tensor):
    """An optimization input parameter"""

    _variant = "parameter"


class Variable(_Tensor):
    """An optimization output variable"""

    _variant = "variable"


class _FragmentMethod:
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __get__(self, frag: Any, _objtype=None) -> Callable[..., Any]:
        # This is needed for non-property calls
        if not isinstance(frag, (Model, ModelFragment)):
            raise TypeError(f"Unexpected owner: {frag}")

        def wrapped(*args, **kwargs) -> Any:
            return self(frag, *args, **kwargs)

        return wrapped


_M = TypeVar("_M", contravariant=True)


_R = TypeVar(
    "_R",
    bound=Union[Expression, Quantification, Quantified[Quantifier]],
    covariant=True,
)


class _Aliasable(Protocol[_M, _R]):
    def __call__(self, model: _M, *exprs: ExpressionLike) -> _R:
        pass


@dataclasses.dataclass(frozen=True)
class _Aliased:
    quantifiables: Sequence[Optional[Space]]
    quantifiers: Union[
        None, QuantifierIdentifier, tuple[QuantifierIdentifier, ...]
    ]


class _Alias(Definition, _FragmentMethod):
    label = None

    def __init__(
        self,
        aliasable: _Aliasable[Any, Any],
        name: Name,
        quantifier_names: Optional[Iterable[Name]] = None,
    ):
        super().__init__()
        self._identifier = AliasIdentifier(name=name)
        self._quantifier_names = quantifier_names
        self._aliasable = aliasable
        self._aliased: Optional[_Aliased] = None

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    def render_statement(self, _label: Label, owner: Any) -> Optional[str]:
        _logger.debug("Rendering alias named %s...", self._identifier.name)
        if self._aliased is None:
            return None  # Not used
        quantifiable = tuple(
            q or _integers for q in self._aliased.quantifiables
        )
        outer_domain = domain_from_quantifiable(
            quantifiable, names=self._quantifier_names
        )
        expressions = [Quantifier(q) for q in outer_domain.quantifiers]
        value = self._aliasable(owner, *expressions)
        s = "\\S^a&: "
        with local_formatting_scope(outer_domain.quantifiers):
            if outer_domain.quantifiers:
                s += f"\\forall {outer_domain.render()}, "
            s += render_identifier(self._identifier, *expressions)
            s += " \\doteq "
            if self._aliased.quantifiers is None:
                s += value.render()
            else:
                inner_domain = domain_from_quantifiable(value)
                with local_formatting_scope(inner_domain.quantifiers):
                    if (
                        len(inner_domain.quantifiers) > 1
                        or inner_domain.mask is not None
                    ):
                        s += f"\\{{ {inner_domain.render()} \\}}"
                    else:
                        s += inner_domain.quantifiers[0].space.render()
        return s

    def __call__(self, frag: Any, *subscripts: ExpressionLike) -> Any:
        exprs = tuple(to_expression(s) for s in subscripts)
        if self._aliased is None:
            value = self._aliasable(frag, *exprs)
            if isinstance(value, Expression):
                quantifiers = None
            else:
                quantifiers, _ = unquantify(value)
            self._aliased = _Aliased(
                [expression_space(x) for x in exprs],
                tuple(_quantifier_identifier(q) for q in quantifiers)
                if isinstance(quantifiers, tuple)
                else None
                if quantifiers is None
                else _quantifier_identifier(quantifiers),
            )
        quantifiers = self._aliased.quantifiers
        if quantifiers is None:
            return ExpressionReference(self._identifier, exprs)
        else:
            ref = QuantifiableReference(
                identifier=self._identifier,
                subscripts=exprs,
                quantifiers=quantifiers
                if isinstance(quantifiers, tuple)
                else (quantifiers,),
            )
            if isinstance(self._aliased.quantifiers, tuple):
                return cross(ref)
            else:
                return iter(ref)


def _quantifier_identifier(arg: Any) -> QuantifierIdentifier:
    if not isinstance(arg, Quantifier):
        raise TypeError(f"Space aliases should only return quantifiers: {arg}")
    return arg.identifier


_F = TypeVar("_F", bound=Callable[..., Union[Expression, Quantifiable]])


def alias(
    name: Optional[Name], quantifier_names: Optional[Iterable[Name]] = None
) -> Callable[[_F], _F]:  # TODO: Tighten argument type
    """Decorator promoting a :class:`.Model` method to a named alias

    Args:
        name: The alias' name. If `None`, no alias will be added.
        quantifier_names: Optional names to use for the alias' quantifiers

    The method can return a (potentially quantified) expression or a
    quantification and may accept any number of expression arguments. This is
    useful to make the generated specification more readable by extracting
    commonly used sub-expressions or sub-spaces.

    Finally, the decorated function may be wrapped as a property if doesn't
    have any non-`self` arguments.

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable(products)

            @property
            @alias("t")
            def total_product_count(self):
                return total(self.count(p) for p in self.products)
    """

    def wrap(fn):
        if name is None:
            return fn
        return _Alias(fn, name=name, quantifier_names=quantifier_names)

    return wrap


ConstraintBody = Callable[[_M], Quantified[Predicate]]


class Constraint(Definition, _FragmentMethod):
    """Optimization constraint

    Constraints are best created directly from :class:`.Model` methods via the
    :func:`.constraint` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable(products)

            @constraint
            def at_least_one(self):
                yield total(self.count(p) for p in self.products) >= 1
    """

    identifier = None

    def __init__(
        self,
        body: ConstraintBody,
        label: Optional[Label] = None,
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        if not inspect.isgeneratorfunction(body):
            raise TypeError("Non-generator function constraint body")
        super().__init__()
        self._body = body
        self._label = label
        self.qualifiers = qualifiers

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

    @property
    def label(self):
        return self._label

    def render_statement(self, label: Label, owner: Any) -> Optional[str]:
        _logger.debug("Rendering constraint %s...", label)
        s = f"\\S^c_\\mathrm{{{_render_label(label, self.qualifiers)}}}&: "
        predicate, domain = within_domain(self._body(owner))
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
    qualifiers: Optional[Sequence[Label]] = None,
    disabled=False,
) -> Callable[[ConstraintBody], Constraint]:
    ...


def constraint(
    body: Optional[ConstraintBody] = None,
    *,
    label: Optional[Label] = None,
    qualifiers: Optional[Sequence[Label]] = None,
    disabled=False,
) -> Any:
    """Decorator promoting a :class:`.Model` method to a :class:`.Constraint`

    Args:
        label: Constraint label override. By default the label is derived from
            the method's name.
        qualifiers: Optional list of labels used to qualify the constraint's
            quantifiers. This is useful to override the name of the colums in
            solution dataframes.

    The decorated method should accept only a `self` argument and return a
    quantified :class:`.Predicate`.

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
        if disabled:
            return fn
        return Constraint(fn, label=label, qualifiers=qualifiers)

    return wrap


ObjectiveSense = Literal["max", "min"]
"""Optimization direction"""


ObjectiveBody = Callable[[_M], Expression]
"""Optimization target expression"""


class Objective(Definition, _FragmentMethod):
    """Optimization objective

    Objectives are best created directly from :class:`.Model` methods via the
    :func:`.objective` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable(products)

            @objective
            def minimize_total_count(self):
                return total(self.count(p) for p in self.products)

    """

    identifier = None

    def __init__(
        self,
        body: ObjectiveBody,
        sense: ObjectiveSense,
        label: Optional[Label] = None,
    ):
        super().__init__()
        self._body = body
        self._sense = sense
        self._label = label

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

    def render_statement(self, label: Label, owner: Any) -> Optional[str]:
        _logger.debug("Rendering objective %s...", label)
        sense = self._sense
        if sense is None:
            if label.startswith("min"):
                sense = "min"
            elif label.startswith("max"):
                sense = "max"
            else:
                raise Exception(f"Missing sense for objective {label}")
        expression = to_expression(self._body(owner))
        return f"\\S^o_\\mathrm{{{label}}}&: \\{sense} {expression.render()}"


@overload
def objective(body: ObjectiveBody) -> Objective:
    ...


@overload
def objective(
    *, sense: Optional[ObjectiveSense] = None, label: Optional[Label] = None
) -> Callable[[ObjectiveBody], Objective]:
    ...


def objective(
    body: Optional[ObjectiveBody] = None,
    *,
    sense: Optional[ObjectiveSense] = None,
    label: Optional[Label] = None,
    disabled=False,
) -> Any:
    """Decorator promoting a method to an :class:`.Objective`

    Args:
        sense: Optimization direction. This may be omitted if the method name
            starts with `min` or `max`, in which case the appropriate sense
            will be inferred.
        label: Objective label override. By default the label is derived from
            the method's name.

    The decorated method should accept only a `self` argument and return an
    :class:`.Expression`, which will become the objective's optimization
    target.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            cost = Parameter(products)
            count = Variable(products)

            @objective
            def minimize_cost(self):
                return total(
                    self.count(p) * self.cost(p)
                    for p in self.products
                )

            @objective(sense="max")
            def optimize_count(self):
                return total(self.count(p) for p in self.products)
    """
    if body:
        return Objective(body, sense=_objective_sense(body))

    def wrap(fn):
        if disabled:
            return fn
        method_sense = sense or _objective_sense(fn)
        return Objective(fn, sense=method_sense, label=label)

    return wrap


def _objective_sense(fn: Callable[..., Any]) -> ObjectiveSense:
    name = fn.__name__
    if name.startswith("min"):
        return "min"
    elif name.startswith("max"):
        return "max"
    else:
        raise Exception(f"Missing sense for objective {name}")
