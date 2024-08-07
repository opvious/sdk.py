"""Built-in model fragments

This module exports :class:`~opvious.modeling.ModelFragment` instances for
common use-cases.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ..common import method_decorator, untuple
from .ast import cross, domain, lift, Projection, Quantifiable, total
from .definitions import (
    Constraint,
    Expression,
    ExpressionLike,
    Image,
    Parameter,
    Tensor,
    TensorLike,
    Variable,
    alias,
    constraint,
    interval,
)
from .identifiers import Name
from .model import ModelFragment
from .quantified import Quantified


class MaskedSubspace(ModelFragment):
    """Masked subspace fragment

    Args:
        quantifiables: Underlying quantifiable
        alias_name: Optional name for the masked subset
    """

    def __init__(
        self,
        *quantifiables: Quantifiable,
        alias_name: Optional[Name] = None,
    ) -> None:
        self._alias_name = alias_name
        self._quantifiables = quantifiables
        self._mask = Parameter.indicator(quantifiables)

    @property
    def mask(self) -> Parameter:
        """Parameter controlling the subset's element"""
        return self._mask

    @property
    @alias(lambda init, self: init(self._alias_name))
    def masked(self) -> Quantified:
        """Masked subset

        As a convenience, iterating on the subset directly also yields
        quantifiers from the masked subset.
        """
        for t in cross(self._quantifiables):
            if self._mask(*t):
                yield untuple(t)

    def __iter__(self) -> Iterable[Any]:
        return (untuple(t) for t in cross(self.masked))


class DerivedVariable(ModelFragment):
    """Variable equal to a given equation

    Args:
        body: The equation defining the variable's value
        quantifiables: Variable quantification
        name: Name of the generated variable
        image: Generated variable :class:`~opvious.modeling.Image`
    """

    default_definition = "value"

    def __init__(
        self,
        body: Callable[..., Any],
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Image = Image(),
    ) -> None:
        self._body = body
        self._value = Variable(image, quantifiables, name=name)

    @property
    def value(self) -> Variable:
        return self._value

    @constraint
    def is_defined(self) -> Quantified:
        for t in self._value.space():
            yield self._value(*t) == self._body(*t)

    def __call__(self, *subs: ExpressionLike) -> Expression:
        return self._value(*subs)


@method_decorator(require_call=True)
def derived_variable(
    *quantifiables: Quantifiable,
    name: Optional[Name] = None,
    image: Image = Image(),
) -> Callable[[TensorLike], DerivedVariable]:
    """Transforms a method into a :class:`DerivedVariable` fragment"""

    def wrapper(fn):
        return DerivedVariable(fn, quantifiables, name=name, image=image)

    return wrapper


class MagnitudeVariable(ModelFragment):
    """Absolute value variable fragment

    Args:
        tensor: Non-negative tensor-like
        quantifiables: Tensor quantifiables. Can be omitted if `tensor` is a
            variable or parameter.
        image: Tensor image. Defaults to `tensor`'s if it is a variable or
            parameter, else non-negative reals.
        name: Name of the generated magnitude variable
        projection: Mask used to project the variable's quantification
        lower_bound: Disable the lower bound
        upper_bound: Disable the upper bound

    See also :func:`magnitude_variable` for a decorator equivalent.
    """

    default_definition = "value"

    def __init__(
        self,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Optional[Image] = None,
        projection: Projection = -1,
        lower_bound=True,
        upper_bound=True,
    ) -> None:
        if isinstance(tensor, Tensor):
            if not quantifiables:
                quantifiables = tensor.quantifiables()
            if not image:
                image = tensor.image
            if lower_bound and tensor.image.lower_bound == 0:
                lower_bound = False
            if upper_bound and tensor.image.upper_bound == 0:
                upper_bound = False
        self._tensor = tensor
        self._domains = tuple(domain(q) for q in quantifiables)
        self._projection = projection
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._value = Variable(
            image or Image(lower_bound=0),
            self._quantification(),
            name=name,
        )

    def _quantification(self, lift=False):
        return cross(*self._domains, projection=self._projection, lift=lift)

    @property
    def value(self) -> Variable:
        """The magnitude variable"""
        return self._value

    @constraint(lambda init, self: init(disabled=not self._lower_bound))
    def lower_bounds(self) -> Quantified:
        """The magnitude's lower bound constraint"""
        for cp in self._quantification(lift=True):
            yield -self.value(*cp) <= self._tensor(*cp.lifted)

    @constraint(lambda init, self: init(disabled=not self._upper_bound))
    def upper_bounds(self) -> Quantified:
        """The magnitude's upper bound constraint"""
        for cp in self._quantification(lift=True):
            yield self._value(*cp) >= self._tensor(*cp.lifted)

    def __call__(self, *subs: ExpressionLike) -> Expression:
        return self._value(*subs)


@method_decorator()
def magnitude_variable(
    *quantifiables: Quantifiable,
    name: Optional[Name] = None,
    image: Optional[Image] = None,
    projection: Projection = -1,
    lower_bound=True,
    upper_bound=True,
) -> Callable[[Callable[..., TensorLike]], MagnitudeVariable]:
    """Transforms a method into a :class:`MagnitudeVariable` fragment

    Note that this method may alter the underlying method's call signature if a
    projection is specified.
    """

    def wrapper(fn):
        return MagnitudeVariable(
            fn,
            *quantifiables,
            name=name,
            image=image,
            projection=projection,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    return wrapper


class ActivationVariable(ModelFragment):
    """Indicator variable activation fragment

    This variable tracks an underlying tensor or tensor-like expression.
    Assuming both of its bounds are defined (see below) it will be equal to 1
    iff the underlying tensor is positive and 0 otherwise.

    Args:
        tensor: Tensor-like
        quantifiables: Tensor quantifiables, can be omitted if the tensor is a
            variable or parameter
        upper_bound: Value of the upper bound used in the activation
            constraint. If `True` the variable's image's upper bound will be
            used, if `False` no activation constraint will be added.
        lower_bound: Value of the lower bound used in the deactivation
            constraint. If `True` the variable's image's lower bound will be
            used, if `False` no deactivation constraint will be added.
        name: Name of the generated activation variable
        negate: Negate the returned indicator variable.
        projection: Mask used to project the variable's quantification. When
            this is set, the indicator variable will be set to 1 iff at least
            one of the projected tensor values is positive.
    """

    # This fragment shows an alternate implementation which does not rely on
    # lazy method decorators. It uses a closure instead of instance variables.
    def __new__(
        cls,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        upper_bound: Union[ExpressionLike, TensorLike, bool] = True,
        lower_bound: Union[ExpressionLike, TensorLike, bool] = False,
        name: Optional[Name] = None,
        negate=False,
        projection: Projection = -1,
    ) -> ActivationVariable:
        if not quantifiables and isinstance(tensor, Tensor):
            quantifiables = tensor.quantifiables()
        domains = tuple(domain(q) for q in quantifiables)

        def quantification(lift=False, projection=projection):
            return cross(*domains, projection=projection, lift=lift)

        def tensor_image():
            if not isinstance(tensor, Tensor):
                raise ValueError(
                    f"Cannot infer bound for tensor-like {tensor}"
                )
            return tensor.image

        class _Fragment(ActivationVariable):
            value = Variable.indicator(quantification(), name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

            @constraint(disabled=upper_bound is False)
            def activates(self):
                bound = upper_bound
                for cp in quantification(lift=True):
                    if callable(bound):
                        bound = bound(*cp.lifted)
                    elif bound is True:
                        bound = tensor_image().upper_bound
                    value = 1 - self.value(*cp) if negate else self.value(*cp)
                    yield bound * value >= tensor(*cp.lifted)

            @constraint(disabled=lower_bound is False)
            def deactivates(self):
                bound = lower_bound
                for cp in quantification():
                    if projection >= 0:
                        term = total(
                            tensor(*lift(cp, ep, projection))
                            for ep in quantification(projection=~projection)
                        )
                    else:
                        term = tensor(*cp)
                    if callable(bound):
                        bound = bound(*cp)
                    elif bound is True:
                        bound = tensor_image().lower_bound
                    value = 1 - self.value(*cp) if negate else self.value(*cp)
                    yield bound * value <= term

        return _Fragment()

    default_definition = "value"

    @property
    def value(self) -> Variable:
        """Activation variable value

        As a convenience calling the fragment directly also returns expressions
        from this variable.
        """
        raise NotImplementedError()

    @property
    def activates(self) -> Optional[Constraint]:
        """Constraint ensuring that the activation variable activates

        This constraint enforces that the activation variable is set to 1 when
        at least one the underlying tensor's value is positive.
        """
        raise NotImplementedError()

    @property
    def deactivates(self) -> Optional[Constraint]:
        """Constraint ensuring that the activation variable deactivates

        This constraint enforces that the activation variable is set to 0 when
        none of the underlying tensor's values are non-zero. It requires the
        fragment to have a non-zero lower bound.
        """
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()


@method_decorator(require_call=True)
def activation_variable(
    *quantifiables: Quantifiable,
    upper_bound: Union[ExpressionLike, TensorLike, bool] = True,
    lower_bound: Union[ExpressionLike, TensorLike, bool] = False,
    name: Optional[Name] = None,
    negate=False,
    projection: Projection = -1,
) -> Callable[[Callable[..., TensorLike]], ActivationVariable]:
    """Transforms a method into an :class:`ActivationVariable` fragment

    Note that this method may alter the underlying method's call signature if a
    projection is specified. See :class:`ActivationVariable` for argument
    documentation.
    """

    def wrapper(fn):
        return ActivationVariable(
            fn,
            *quantifiables,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            name=name,
            negate=negate,
            projection=projection,
        )

    return wrapper


class PiecewiseLinear(ModelFragment):
    """Multiplication with a piecewise-linear factor

    Args:
        tensor: Tensor-like
        quantifiables: Underlying quantifiable
        assume_convex: Assume that the factors are increasing
        component_name: Name of the generated variable used to represent the
            value in each segmented piece.
        pieces_name: Name of the interval representing all piece indices.
        piece_count_name: Name of the generated piece count parameter.
        factor_name: Name of the generated factor parameter.
        width_name: Name of the generated width parameter.
    """

    def __init__(
        self,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        assume_convex=False,
        pieces_name: Optional[str] = None,
        piece_count_name: Optional[str] = None,
        component_name: Optional[str] = None,
        factor_name: Optional[str] = None,
        width_name: Optional[str] = None,
    ) -> None:
        if not assume_convex:
            raise NotImplementedError()  # TODO: Implement.

        self._tensor = tensor
        if not quantifiables and isinstance(tensor, Tensor):
            quantifiables = tensor.quantifiables()
        self._domains = tuple(domain(q) for q in quantifiables)

        self._piece_count = Parameter.discrete(
            lower_bound=1, name=piece_count_name
        )
        self._pieces = interval(1, self.piece_count(), name=pieces_name)
        self._component = Variable.non_negative(
            self._pieces, self._domains, name=component_name
        )
        self._factor = Parameter.continuous(self._pieces, name=factor_name)
        self._width = Parameter.continuous(self._pieces, name=width_name)

    @property
    def piece_count(self) -> Parameter:
        """The total number of pieces"""
        return self._piece_count

    @property
    def pieces(self):
        return self._pieces

    @property
    def factor(self) -> Parameter:
        """The factor to multiply each component with"""
        return self._factor

    @property
    def width(self) -> Parameter:
        """The width of each segment for the resulting variable"""
        return self._width

    @property
    def component(self) -> Variable:
        """The underlying segmented variable"""
        return self._component

    @constraint
    def component_is_within_width(self) -> Quantified:
        """Asserts that each component is below its width"""
        for p in self._pieces:
            for tp in cross(self._domains):
                yield self._component(p, *tp) <= self._width(p)

    @constraint
    def component_total_matches_tensor(self) -> Quantified:
        """Asserts that the sum of the components matches the input tensor"""
        for tp in cross(self._domains):
            yield total(
                self._component(p, *tp) for p in self._pieces
            ) == self._tensor(*tp)

    def __call__(self, *subs: ExpressionLike) -> Expression:
        return total(
            self._component(p, *subs) * self._factor(p) for p in self._pieces
        )

    def total(self):
        """Returns the fully quantified piecewise-linear sum"""
        return total(
            self._component(*tp) * self._factor(tp[0])
            for tp in cross(self._pieces, self._domains)
        )


@method_decorator(require_call=True)
def piecewise_linear(
    *quantifiables: Quantifiable,
    assume_convex=False,
    pieces_name: Optional[str] = None,
    piece_count_name: Optional[str] = None,
    component_name: Optional[str] = None,
    factor_name: Optional[str] = None,
    width_name: Optional[str] = None,
) -> Callable[[Callable[..., TensorLike]], PiecewiseLinear]:
    """Transforms a method into an :class:`PiecewiseLinear` fragment

    See :class:`PiecewiseLinear` for argument documentation.
    """

    def wrapper(fn):
        return PiecewiseLinear(
            fn,
            *quantifiables,
            assume_convex=assume_convex,
            pieces_name=pieces_name,
            piece_count_name=piece_count_name,
            component_name=component_name,
            factor_name=factor_name,
            width_name=width_name,
        )

    return wrapper
