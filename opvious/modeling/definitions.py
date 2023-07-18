from __future__ import annotations

import dataclasses
import functools
import inspect
import itertools
import logging
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from ..common import Label
from .ast import (
    Domain,
    Expression,
    ExpressionLike,
    ExpressionReference,
    IterableSpace,
    Predicate,
    Quantifiable,
    QuantifiableReference,
    Quantifier,
    QuantifierIdentifier,
    ScalarSpace,
    Space,
    cross,
    domain,
    expression_space,
    is_literal,
    literal,
    render_identifier,
    to_expression,
    total,
    within_domain,
)
from .identifiers import (
    AliasIdentifier,
    DimensionIdentifier,
    GlobalIdentifier,
    Name,
    TensorIdentifier,
    local_formatting_scope,
)
from .quantified import Quantified, unquantify
from .statements import Definition, Model, ModelFragment, method_decorator


_logger = logging.getLogger(__name__)


class Dimension(Definition, ScalarSpace):
    """An abstract collection of values

    Args:
        label: Dimension label override. By default the label is derived from
            the attribute's name
        name: The dimension's name. By default the name will be derived from
            the dimension's label
        is_numeric: Whether the dimension will only contain integers. This
            enables arithmetic operations on this dimension's quantifiers

    Dimensions are `Quantifiable` and as such can be quantified over using
    :func:`.cross`. As a convenience, iterating on a dimension also yields a
    suitable quantifier. This allows creating simple constraints directly:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable.natural(products)

            @constraint
            def at_least_one_of_each(self):
                for p in self.products:  # Note the iteration here
                    yield self.count(p) >= 1
    """

    category = "DIMENSION"

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

    def render_statement(self, label: Label) -> Optional[str]:
        _logger.debug("Rendering dimension %s...", label)
        s = f"\\S^d_\\mathrm{{{label}}}&: {self._identifier.format()}"
        if self._is_numeric:
            s += " \\subset \\mathbb{Z}"
        return s


@dataclasses.dataclass(frozen=True)
class _Interval(ScalarSpace):
    lower_bound: Expression
    upper_bound: Expression

    def render(self) -> str:
        lb = self.lower_bound
        ub = self.upper_bound
        if is_literal(lb, 0) and is_literal(ub, 1):
            return "\\{0, 1\\}"
        if is_literal(ub, math.inf):
            if is_literal(lb, 0):
                return "\\mathbb{N}"
            elif is_literal(lb, -math.inf):
                return "\\mathbb{Z}"
        return f"\\{{ {lb.render()} \\ldots {ub.render()} \\}}"


_integers = _Interval(literal(-math.inf), literal(math.inf))


def interval(
    lower_bound: ExpressionLike,
    upper_bound: ExpressionLike,
    name: Optional[Name] = None,
) -> IterableSpace[Quantifier]:
    """A range of values

    Args:
        lower_bound: The range's inclusive lower bound
        upper_bound: The range's inclusive upper bound
        name: An optional name to create an alias for this interval
    """
    interval = _Interval(
        lower_bound=to_expression(lower_bound),
        upper_bound=to_expression(upper_bound),
    )

    class _Fragment(ModelFragment, Space):
        @property
        @alias(name)
        def interval(self):
            return interval

        def __iter__(self) -> Iterator[Quantifier]:
            return iter(self.interval)

    return _Fragment()


@dataclasses.dataclass(frozen=True)
class Image:
    """A tensor's set of possible values"""

    lower_bound: ExpressionLike = -math.inf
    """The image's smallest value (inclusive)"""

    upper_bound: ExpressionLike = math.inf
    """The image's largest value (inclusive)"""

    is_integral: bool = False
    """Whether the image only contain integers"""

    def render(self) -> str:
        lb = to_expression(self.lower_bound)
        ub = to_expression(self.upper_bound)
        if is_literal(ub, math.inf):
            if is_literal(lb, -math.inf):
                return "\\mathbb{Z}" if self.is_integral else "\\mathbb{R}"
            if is_literal(lb, 0):
                return "\\mathbb{N}" if self.is_integral else "\\mathbb{R}_+"
        if self.is_integral:
            if is_literal(lb, 0) and is_literal(ub, 1):
                return "\\{0, 1\\}"
            return f"\\{{{lb.render()} \\ldots {ub.render()}\\}}"
        return f"[{lb.render()}, {ub.render()}]"


TensorLike = Callable[..., Expression]


_T = TypeVar("_T", bound="Tensor")


class Tensor(Definition):
    """Base tensor class

    Calling a tensor returns an :class:`~.opvious.modeling.Expression` with any
    arguments as subscripts. For example:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable.natural(products)

            @property
            def total_product_count(self):
                return total(
                    self.count(p)  # Expression representing the count for `p`
                    for p in self.products
                )

    The number of arguments must match the tensor's quantification.
    """

    def __init__(
        self,
        image: Image,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        if not isinstance(image, Image):
            raise TypeError(f"Unexpected image: {image}")
        self._identifier = TensorIdentifier(
            name=name,
            is_parameter=self.category == "PARAMETER",
        )
        # We do not combine domains here to allow quantification projections in
        # an alias-agnostic (and simpler) way
        self._domains = tuple(domain(q) for q in quantifiables)
        self._label = label
        self._image = image
        self.qualifiers = qualifiers

    @classmethod
    def continuous(cls: Type[_T], *quantifiables, **kwargs) -> _T:
        """Returns a tensor with real image"""
        return cls(Image(), *quantifiables, **kwargs)

    @classmethod
    def non_negative(
        cls: Type[_T],
        *quantifiables,
        upper_bound: ExpressionLike = math.inf,
        **kwargs,
    ) -> _T:
        """Returns a tensor with non-negative real image"""
        img = Image(lower_bound=0, upper_bound=upper_bound)
        return cls(img, *quantifiables, **kwargs)

    @classmethod
    def non_positive(cls: Type[_T], *quantifiables, **kwargs) -> _T:
        """Returns a tensor with non-positive real image"""
        return cls(Image(upper_bound=0), *quantifiables, **kwargs)

    @classmethod
    def unit(cls: Type[_T], *quantifiables, **kwargs) -> _T:
        """Returns a tensor with `[0, 1]` real image"""
        return cls(
            Image(lower_bound=0, upper_bound=1), *quantifiables, **kwargs
        )

    @classmethod
    def discrete(cls: Type[_T], *quantifiables, **kwargs) -> _T:
        """Returns a tensor with integral image"""
        return cls(Image(is_integral=True), *quantifiables, **kwargs)

    @classmethod
    def natural(
        cls: Type[_T],
        *quantifiables,
        upper_bound: ExpressionLike = math.inf,
        **kwargs,
    ) -> _T:
        """Returns a tensor with non-negative integral image"""
        img = Image(lower_bound=0, upper_bound=upper_bound, is_integral=True)
        return cls(img, *quantifiables, **kwargs)

    @classmethod
    def indicator(cls: Type[_T], *quantifiables, **kwargs) -> _T:
        """Returns a tensor with `{0, 1}` integral image"""
        image = Image(lower_bound=0, upper_bound=1, is_integral=True)
        return cls(image, *quantifiables, **kwargs)

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._label

    @property
    def image(self) -> Image:
        """The tensor's values' image"""
        return self._image

    def quantifiables(self) -> tuple[Quantifiable, ...]:
        """The quantifiables generating the tensor's space

        This is typically useful to support projections in model fragments. For
        most use cases consider using the simpler
        :meth:`~opvious.modeling.Tensor.space` method.
        """
        return self._domains

    def space(self) -> IterableSpace[Sequence[Quantifier]]:
        """Returns the tensor's underlying space"""
        return cross(self._domains)

    def __call__(self, *subscripts: ExpressionLike) -> Expression:
        return ExpressionReference(
            self._identifier, tuple(to_expression(s) for s in subscripts)
        )

    def total(self, absolute=False) -> Expression:
        """The tensor's total summed value

        Args:
            absolute: Sum the absolute value of the tensor's terms instead of
                its raw values
        """
        return total(
            abs(self(*q)) if absolute else self(*q) for q in self.space()
        )

    def render_statement(self, label: Label) -> Optional[str]:
        _logger.debug("Rendering tensor %s...", label)
        c = self.category[0].lower()
        s = f"\\S^{c}_\\mathrm{{{_render_label(label, self.qualifiers)}}}&: "
        s += f"{self._identifier.format()} \\in "
        s += self.image.render()
        d = domain(self._domains)
        if d.quantifiers:
            with local_formatting_scope(d.quantifiers):
                if _is_simple_domain(d):
                    formatted: list[str] = []
                    for g, qs in itertools.groupby(
                        d.quantifiers, key=lambda q: q.outer_group
                    ):
                        if g is None:
                            formatted.extend(q.space.render() for q in qs)
                        else:
                            formatted.append(g.alias.format())
                    sup = " \\times ".join(formatted)
                else:
                    sup = f"\\{{ {d.render()} \\}}"
                s += f"^{{{sup}}}"
        return s


def _is_simple_domain(d: Domain) -> bool:
    if d.mask is not None:
        return False
    for q in d.quantifiers:
        for g in q.groups:
            if g.subscripts:
                return False
    return True


def _render_label(label: Label, qualifiers: Optional[Sequence[Label]]) -> str:
    s = label
    if qualifiers:
        s += f"[{','.join(qualifiers)}]"
    return s


class Parameter(Tensor):
    """An optimization input parameter

    Args:
        image: Target :class:`~opvious.modeling.Image`
        quantifiables: Quantification
        label: Optional label, if unset it will be derived from the parameter's
            attribute name
        name: Optional name, if unset it will be derived from the labe
        qualifiers: Optional list of labels used to name this parameter's
            quantifiables

    Consider instantiating parameters via one of the various
    :class:`~opvious.modeling.Tensor` convenience class methods, for example:

    .. code-block:: python

        p1 = Parameter.continuous()  # Real-valued parameter
        p2 = Parameter.natural() # # Parameter with values in {0, 1...}

    Each parameter will be expected to have a matching input when solving this
    model on data.
    """

    category = "PARAMETER"


class Variable(Tensor):
    """An optimization output variable

    Args:
        image: Target :class:`~opvious.modeling.Image`
        quantifiables: Quantification
        label: Optional label, if unset it will be derived from the parameter's
            attribute name
        name: Optional name, if unset it will be derived from the labe
        qualifiers: Optional list of labels used to name this parameter's
            quantifiables

    Similar to parameters, consider instantiating variables via one of the
    various :class:`~opvious.modeling.Tensor` convenience class methods, for
    example:

    .. code-block:: python

        v1 = Variable.unit()  # Variable with value within [0, 1]
        v2 = Variable.non_negative() # # Variable with value at least 0

    Each variable will have its results available in feasible solve solutions.
    """

    category = "VARIABLE"


_Aliasable = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class _Aliased:
    quantifiable: Quantifiable
    quantifiers: Union[
        None, QuantifierIdentifier, tuple[QuantifierIdentifier, ...]
    ]


class _Alias(Definition):
    category = "ALIAS"
    label = None

    def __init__(
        self,
        aliasable: _Aliasable,
        quantifiable: tuple[Quantifiable],
        name: Name,
        quantifier_names: Optional[Iterable[Name]] = None,
    ):
        super().__init__()
        self._identifier = AliasIdentifier(name=name)
        self._quantifier_names = quantifier_names
        self._aliasable = aliasable
        self._aliased: Optional[_Aliased] = None
        self._quantifiable = quantifiable

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    def render_statement(self, _label: Label) -> Optional[str]:
        if self._aliased is None:
            _logger.debug("Skipping alias named %s.", self._identifier.name)
            return None  # Not used

        _logger.debug("Rendering alias named %s...", self._identifier.name)
        outer_domain = domain(
            self._aliased.quantifiable, names=self._quantifier_names
        )
        expressions = [Quantifier(q) for q in outer_domain.quantifiers]
        value = self._aliasable(*expressions)

        s = "\\S^a&: "
        with local_formatting_scope(outer_domain.quantifiers):
            if outer_domain.quantifiers:
                s += f"\\forall {outer_domain.render()}, "
            s += render_identifier(self._identifier, *expressions)
            s += " \\doteq "
            if self._aliased.quantifiers is None:
                s += value.render()
            else:
                inner_domain = domain(value)
                with local_formatting_scope(inner_domain.quantifiers):
                    if (
                        len(inner_domain.quantifiers) > 1
                        or inner_domain.mask is not None
                    ):
                        s += f"\\{{ {inner_domain.render()} \\}}"
                    else:
                        s += inner_domain.quantifiers[0].space.render()
        return s

    def __call__(self, *subs: ExpressionLike) -> Any:
        exprs = tuple(to_expression(s) for s in subs)
        if self._aliased is None:
            value = self._aliasable(*exprs)
            if isinstance(value, Expression):
                quantifiers = None
            else:
                quantifiers, _ = unquantify(value)
            self._aliased = _Aliased(
                self._quantifiable
                or tuple(expression_space(x) or _integers for x in exprs),
                None
                if quantifiers is None
                else _quantifier_identifier(quantifiers)
                if isinstance(quantifiers, Quantifier)
                else tuple(_quantifier_identifier(q) for q in quantifiers),
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


_M = TypeVar("_M", bound=Union[Model, ModelFragment], contravariant=True)


_F = TypeVar("_F", bound=Callable[..., Union[Expression, Quantifiable]])


def alias(
    name: Optional[Name],
    *quantifiables: Quantifiable,
    quantifier_names: Optional[Iterable[Name]] = None,
) -> Callable[[_F], _F]:  # TODO: Tighten argument type
    """Decorator promoting a :class:`.Model` method to a named alias

    Args:
        name: The alias' name. If `None`, no alias will be added.
        quantifiables: The alias' quantification. If omitted, it will be
            inferred from the alias' calls.
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
            count = Variable.natural(products)

            @property
            @alias("t")
            def total_product_count(self):
                return total(self.count(p) for p in self.products)
    """

    @method_decorator
    def wrapper(fn):
        if name is None:
            return fn
        return _Alias(
            fn,
            quantifiable=quantifiables,
            name=name,
            quantifier_names=quantifier_names,
        )

    return wrapper


ConstraintMethod = Callable[[_M], Quantified[Predicate]]


class Constraint(Definition):
    """Optimization constraint

    Constraints are best created directly from :class:`.Model` methods via the
    :func:`.constraint` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable.natural(products)

            @constraint
            def at_least_one(self):
                yield total(self.count(p) for p in self.products) >= 1
    """

    category = "CONSTRAINT"
    identifier = None

    def __init__(
        self,
        body: Callable[[], Quantified[Predicate]],
        label: Optional[Label] = None,
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        super().__init__()
        self._body = body
        self._label = label
        self.qualifiers = qualifiers

    @property
    def label(self):
        return self._label

    def render_statement(self, label: Label) -> Optional[str]:
        _logger.debug("Rendering constraint %s...", label)

        s = f"\\S^c_\\mathrm{{{_render_label(label, self.qualifiers)}}}&: "
        predicate, d = within_domain(self._body())
        with local_formatting_scope(d.quantifiers):
            if d.quantifiers:
                s += f"\\forall {d.render()}, "
            s += predicate.render()
        return s


@overload
def constraint(method: ConstraintMethod) -> Constraint:
    ...


@overload
def constraint(
    *,
    label: Optional[Label] = None,
    qualifiers: Optional[Sequence[Label]] = None,
    disabled=False,
) -> Callable[[ConstraintMethod], Optional[Constraint]]:
    ...


def constraint(
    method: Optional[ConstraintMethod] = None,
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
        disabled: Disable the constraint

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

    @method_decorator
    def wrapper(fn):
        if not inspect.isgeneratorfunction(fn):
            raise TypeError(f"Non-generator constraint function: {fn}")
        if disabled:
            return None
        return Constraint(fn, label=label, qualifiers=qualifiers)

    return wrapper(method) if method else wrapper


ObjectiveSense = Literal["max", "min"]
"""Optimization direction"""


ObjectiveMethod = Callable[[_M], Expression]
"""Optimization target expression"""


class Objective(Definition):
    """Optimization objective

    Objectives are best created directly from :class:`.Model` methods via the
    :func:`.objective` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable.natural(products)

            @objective
            def minimize_total_count(self):
                return total(self.count(p) for p in self.products)

    """

    category = "OBJECTIVE"
    identifier = None

    def __init__(
        self,
        body: Callable[[], Expression],
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

    def render_statement(self, label: Label) -> Optional[str]:
        _logger.debug("Rendering objective %s...", label)

        sense = self._sense
        if sense is None:
            if label.startswith("min"):
                sense = "min"
            elif label.startswith("max"):
                sense = "max"
            else:
                raise Exception(f"Missing sense for objective {label}")

        expression = to_expression(self._body())
        return f"\\S^o_\\mathrm{{{label}}}&: \\{sense} {expression.render()}"


@overload
def objective(method: ObjectiveMethod) -> Objective:
    ...


@overload
def objective(
    *,
    sense: Optional[ObjectiveSense] = None,
    label: Optional[Label] = None,
    disabled=False,
) -> Callable[[ObjectiveMethod], Objective]:
    ...


def objective(
    method: Optional[ObjectiveMethod] = None,
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
        disabled: Disable the objective

    The decorated method should accept only a `self` argument and return an
    :class:`.Expression`, which will become the objective's optimization
    target.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            cost = Parameter.non_negative(products)
            count = Variable.natural(products)

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

    @method_decorator
    def wrapper(fn):
        if disabled:
            return None
        method_sense = sense or _objective_sense(fn)
        return Objective(fn, sense=method_sense, label=label)

    return wrapper(method) if method else wrapper


def _objective_sense(fn: Callable[..., Any]) -> ObjectiveSense:
    while isinstance(fn, functools.partial):
        fn = fn.func
    name = fn.__name__
    if name.startswith("min"):
        return "min"
    elif name.startswith("max"):
        return "max"
    else:
        raise Exception(f"Missing sense for objective {name}")
