from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ...common import untuple
from .ast import cross, Quantifiable
from .definitions import (
    Constraint,
    Expression,
    ExpressionLike,
    Image,
    Parameter,
    Variable,
    alias,
    constraint,
)
from .identifiers import Name
from .quantified import Quantified
from .statements import method_decorator, ModelFragment


class ActivationIndicator(ModelFragment):
    """Variable activation tracking"""

    def __new__(
        cls,
        variable: Variable,
        *,
        upper_bound: Union[ExpressionLike, bool] = False,
        lower_bound: Union[ExpressionLike, bool] = False,
        name: Optional[Name] = None,
    ) -> ActivationIndicator:
        """Returns a variable activation fragment

        Args:
            variable: Non-negative variable
            upper_bound: Value of the upper bound used in the activation
                constraint. If `True` the variable's image's upper bound will
                be used, if `False` no activation constraint will be added.
            lower_bound: Value of the lower bound used in the deactivation
                constraint. If `True` the variable's image's lower bound will
                be used, if `False` no deactivation constraint will be added.
            name: Name of the generated activation variable
        """

        class _Fragment(ActivationIndicator):
            value = Variable.indicator(variable.quantification, name=name)

            def __new__(cls) -> _Fragment:
                return ModelFragment.__new__(cls)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

            @constraint(disabled=upper_bound is False)
            def activates(self):
                bound = upper_bound
                if bound is True:
                    bound = variable.image.upper_bound
                for t in variable.quantification:
                    yield variable(*t) <= bound * self.value(*t)

            @constraint(disabled=lower_bound is False)
            def deactivates(self):
                bound = lower_bound
                if bound is True:
                    bound = variable.image.lower_bound
                for t in variable.quantification:
                    yield variable(*t) >= bound * self.value(*t)

        return _Fragment()

    @property
    def value(self) -> Variable:
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
        raise NotImplementedError()

    @property
    def masked(self) -> Quantified:
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
                for t in cross(self.value.quantification):
                    yield self.value(*t) == body(*t)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

        return _Fragment()

    @property
    def value(self) -> Variable:
        raise NotImplementedError()

    @property
    def is_defined(self) -> Constraint:
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
