"""Built-in model fragments

This module exports :class:`~opvious.modeling.ModelFragment` instances for
common use-cases. As a convenience it is also exported by the
`opvious.modeling` module.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ..common import untuple
from .ast import cross, domain, Projection, Quantifiable
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


class Activation(ModelFragment):
    """Variable activation tracking"""

    def __new__(
        cls,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        upper_bound: Union[ExpressionLike, bool] = False,
        lower_bound: Union[ExpressionLike, bool] = False,
        name: Optional[Name] = None,
        projection: Projection = -1,
    ) -> Activation:
        """Returns a variable activation fragment

        Args:
            tensor: Non-negative tensor-like
            quantifiables: Tensor quantifiables, can be omitted if the tensor
                is a variable or parameter
            upper_bound: Value of the upper bound used in the activation
                constraint. If `True` the variable's image's upper bound will
                be used, if `False` no activation constraint will be added.
            lower_bound: Value of the lower bound used in the deactivation
                constraint. If `True` the variable's image's lower bound will
                be used, if `False` no deactivation constraint will be added.
            name: Name of the generated activation variable
            projection: Mask used to project the variable's quantification
        """
        if not quantifiables and isinstance(tensor, Tensor):
            quantifiables = tensor.quantifiables()
        domains = tuple(domain(q) for q in quantifiables)

        def quantification(lift=False):
            return cross(domains, projection=projection, lift=lift)

        def tensor_image():
            if not isinstance(tensor, Tensor):
                raise ValueError(
                    f"Cannot infer bound for tensor-like {tensor}"
                )
            return tensor.image

        class _Fragment(Activation):
            value = Variable.indicator(quantification(), name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

            @constraint(disabled=upper_bound is False)
            def activates(self):
                bound = upper_bound
                if bound is True:
                    bound = tensor_image().upper_bound
                for cp in quantification(lift=True):
                    yield tensor(*cp.lifted) <= bound * self.value(*cp)

            @constraint(disabled=lower_bound is False)
            def deactivates(self):
                bound = lower_bound
                if bound is True:
                    bound = tensor_image().lower_bound
                for cp in quantification(lift=True):
                    yield tensor(*cp.lifted) >= bound * self.value(*cp)

        return _Fragment()

    @property
    def value(self) -> Variable:
        raise NotImplementedError()

    @property
    def activates(self) -> Optional[Constraint]:
        raise NotImplementedError()

    @property
    def deactivates(self) -> Optional[Constraint]:
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()


class MaskedSubset(ModelFragment):
    """Quantifiable subset"""

    def __new__(
        cls,
        *quantifiables: Quantifiable,
        alias_name: Optional[Name] = None,
    ) -> MaskedSubset:
        class _Fragment(MaskedSubset):
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
        """The parameter controlling the subset's element"""
        raise NotImplementedError()

    @property
    def masked(self) -> Quantified:
        """The masked subset"""
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Any]:
        raise NotImplementedError()


class DerivedVariable(ModelFragment):
    """Variable equal to a given equation"""

    def __new__(
        cls,
        body: Callable[..., Any],
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Image = Image(),
    ) -> DerivedVariable:
        """Returns a derived variable fragment"""

        class _Fragment(DerivedVariable):
            value = Variable(image, quantifiables, name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            @constraint
            def is_defined(self) -> Quantified:
                for t in cross(self.value.quantifiables()):
                    yield self.value(*t) == body(*t)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

        return _Fragment()

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
) -> Callable[[Callable[..., Expression]], DerivedVariable]:
    """Transforms a method into a derived variable"""

    @method_decorator
    def wrapper(fn):
        return DerivedVariable(fn, quantifiables, name=name, image=image)

    return wrapper


class Magnitude(ModelFragment):
    """Absolute value variable fragment"""

    def __new__(
        cls,
        tensor: TensorLike,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        projection: Projection = -1,
    ) -> Magnitude:
        if not quantifiables and isinstance(tensor, Tensor):
            quantifiables = tensor.quantifiables()
        domains = tuple(domain(q) for q in quantifiables)

        def quantification(lift=False):
            return cross(domains, projection=projection, lift=lift)

        class _Fragment(Magnitude):
            value = Variable.non_negative(quantification(), name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

            @constraint
            def lower_bounds(self):
                for cp in quantification(lift=True):
                    yield tensor(*cp.lifted) >= -self.value(*cp)

            @constraint
            def upper_bounds(self):
                for cp in quantification(lift=True):
                    yield tensor(*cp.lifted) <= self.value(*cp)

        return _Fragment()

    @property
    def value(self) -> Variable:
        """The magnitude variable"""
        raise NotImplementedError()

    @property
    def lower_bounds(self) -> Constraint:
        """The magnitude's lower bound constraint"""
        raise NotImplementedError()

    @property
    def upper_bounds(self) -> Constraint:
        """The magnitude's upper bound constraint"""
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()
