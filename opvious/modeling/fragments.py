"""Built-in model fragments

This module exports :class:`~opvious.modeling.ModelFragment` instances for
common use-cases. As a convenience it is also exported by the
`opvious.modeling` module.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union, cast

from ..common import untuple
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
)
from .identifiers import Name
from .quantified import Quantified
from .statements import method_decorator, ModelFragment


class MaskedSubspace(ModelFragment):
    """Masked subspace fragment"""

    def __new__(
        cls,
        *quantifiables: Quantifiable,
        alias_name: Optional[Name] = None,
    ) -> MaskedSubspace:
        class _Fragment(MaskedSubspace):
            mask = Parameter.indicator(quantifiables)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            @property
            @alias(alias_name)
            def masked(self) -> Quantified:
                for t in cross(quantifiables):
                    if self.mask(*t):
                        yield untuple(t)

            def __iter__(self):
                return (untuple(t) for t in cross(self.masked))

        return _Fragment()

    @property
    def mask(self) -> Parameter:
        """Parameter controlling the subset's element"""
        raise NotImplementedError()

    @property
    def masked(self) -> Quantified:
        """Masked subset

        As a convenience, iterating on the subset directly also yields
        quantifiers from the masked subset.
        """
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Any]:
        raise NotImplementedError()


MaskedSubset = MaskedSubspace  # Deprecated alias


class DerivedVariable(ModelFragment):
    """Variable equal to a given equation

    Args:
        body: The equation defining the variable's value
        quantifiables: Variable quantification
        name: Name of the generated variable
        image: Generated variable :class:`~opvious.modeling.Image`
    """

    def __new__(
        cls,
        body: Callable[..., Any],
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Image = Image(),
    ) -> DerivedVariable:
        class _Fragment(DerivedVariable):
            value = Variable(image, quantifiables, name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            @constraint
            def is_defined(self) -> Quantified:
                for t in self.value.space():
                    yield self.value(*t) == body(*t)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

        return _Fragment()

    default_definition = "value"

    @property
    def value(self) -> Variable:
        """The generated variable"""
        raise NotImplementedError()

    @property
    def is_defined(self) -> Constraint:
        """The constraint ensuring the variable's value"""
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()


def derived_variable(
    *quantifiables: Quantifiable,
    name: Optional[Name] = None,
    image: Image = Image(),
) -> Callable[[TensorLike], DerivedVariable]:
    """Transforms a method into a :class:`DerivedVariable` fragment"""

    @method_decorator
    def wrapper(fn):
        return DerivedVariable(fn, quantifiables, name=name, image=image)

    return wrapper


class ActivationVariable(ModelFragment):
    """Indicator variable activation fragment

    Args:
        tensor: Non-negative tensor-like
        quantifiables: Tensor quantifiables, can be omitted if the tensor is a
            variable or parameter
        upper_bound: Value of the upper bound used in the activation
            constraint. If `True` the variable's image's upper bound will be
            used, if `False` no activation constraint will be added.
        lower_bound: Value of the lower bound used in the deactivation
            constraint. If `True` the variable's image's lower bound will be
            used, if `False` no deactivation constraint will be added.
        name: Name of the generated activation variable
        projection: Mask used to project the variable's quantification
    """

    def __new__(
        cls,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        upper_bound: Union[ExpressionLike, TensorLike, bool] = True,
        lower_bound: Union[ExpressionLike, TensorLike, bool] = False,
        name: Optional[Name] = None,
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
                    yield bound * self.value(*cp) >= tensor(*cp.lifted)

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
                    yield bound * self.value(*cp) <= term

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


ActivationIndicator = ActivationVariable  # Deprecated alias


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

    def __new__(
        cls,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Optional[Image] = None,
        projection: Projection = -1,
        lower_bound=True,
        upper_bound=True,
    ) -> MagnitudeVariable:
        if isinstance(tensor, Tensor):
            if not quantifiables:
                quantifiables = tensor.quantifiables()
            if not image:
                image = tensor.image
            if lower_bound and tensor.image.lower_bound == 0:
                lower_bound = False
            if upper_bound and tensor.image.upper_bound == 0:
                upper_bound = False
        domains = tuple(domain(q) for q in quantifiables)
        if image is None:
            image = Image(lower_bound=0)

        def quantification(lift=False):
            return cross(*domains, projection=projection, lift=lift)

        class _Fragment(MagnitudeVariable):
            value = Variable(cast(Any, image), quantification(), name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

            @constraint(disabled=not lower_bound)
            def lower_bounds(self):
                for cp in quantification(lift=True):
                    yield -self.value(*cp) <= tensor(*cp.lifted)

            @constraint(disabled=not upper_bound)
            def upper_bounds(self):
                for cp in quantification(lift=True):
                    yield self.value(*cp) >= tensor(*cp.lifted)

        return _Fragment()

    default_definition = "value"

    @property
    def value(self) -> Variable:
        """The magnitude variable"""
        raise NotImplementedError()

    @property
    def lower_bounds(self) -> Optional[Constraint]:
        """The magnitude's lower bound constraint"""
        raise NotImplementedError()

    @property
    def upper_bounds(self) -> Optional[Constraint]:
        """The magnitude's upper bound constraint"""
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()


def magnitude_variable(
    *quantifiables: Quantifiable,
    name: Optional[Name] = None,
    image: Optional[Image] = None,
    projection: Projection = -1,
    lower_bound=True,
    upper_bound=True,
) -> Callable[[TensorLike], MagnitudeVariable]:
    """Transforms a method into a :class:`MagnitudeVariable` fragment

    Note that this method may alter the underlying method's call signature if a
    projection is specified.
    """

    @method_decorator
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


Magnitude = MagnitudeVariable  # Deprecated alias
