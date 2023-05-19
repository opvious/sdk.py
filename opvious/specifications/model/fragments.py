from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ...common import untuple
from .ast import cross, Quantifiable
from .definitions import (
    Constraint,
    Expression,
    ExpressionLike,
    ModelFragment,
    Parameter,
    Variable,
    alias,
    constraint,
)
from .identifiers import Name
from .images import Image
from .quantified import Quantified


class ActivationIndicator(ModelFragment):
    """Variable activation tracking"""

    @property
    def is_active(self) -> Variable:
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()

    @classmethod
    def fragment(
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
            is_active = Variable(
                variable.quantification(),
                name=name,
                image=Image.indicator(),
            )

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.is_active(*subs)

            @constraint(disabled=upper_bound is False)
            def activates(self):
                bound = upper_bound
                if bound is True:
                    bound = variable.image.upper_bound
                for t in variable.quantification():
                    yield variable(*t) <= bound * self.is_active(*t)

            @constraint(disabled=lower_bound is False)
            def deactivates(self):
                bound = lower_bound
                if bound is True:
                    bound = variable.image.lower_bound
                for t in variable.quantification():
                    yield variable(*t) >= bound * self.is_active(*t)

        return _Fragment()


class MaskedSubset(ModelFragment):
    """Quantifiable subset"""

    @property
    def mask(self) -> Parameter:
        raise NotImplementedError()

    @property
    def masked(self) -> Quantified:
        raise NotImplementedError()

    def __iter__(self) -> Iterable[Any]:
        raise NotImplementedError()

    @classmethod
    def fragment(
        cls,
        *quantifiables: Quantifiable,
        alias_name: Optional[Name] = None,
    ) -> MaskedSubset:
        """Returns a quantifiable subset fragment"""

        class _Fragment(MaskedSubset):
            mask = Parameter(quantifiables, image=Image.indicator())

            @property
            @alias(alias_name)
            def masked(self) -> Quantified:
                for t in cross(quantifiables):
                    if self.mask(*t):
                        yield untuple(t)

            def __iter__(self):
                return (untuple(t) for t in cross(self.masked))

        return _Fragment()


class DerivedVariable(ModelFragment):
    """Variable equal to a given equation"""

    @property
    def value(self) -> Variable:
        raise NotImplementedError()

    @property
    def is_defined(self) -> Constraint:
        raise NotImplementedError()

    def __call__(self, *subs: ExpressionLike) -> Expression:
        raise NotImplementedError()

    @classmethod
    def fragment(
        cls,
        body: Callable[..., Any],
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        image: Image = Image(),
    ) -> DerivedVariable:
        """Returns a derived variable fragment"""

        class _Fragment(DerivedVariable):
            value = Variable(quantifiables, image=image, name=name)

            @constraint
            def is_defined(self) -> Quantified:
                for t in cross(quantifiables):
                    yield self.value == body(*t)

            def __call__(self, *subs: ExpressionLike) -> Expression:
                return self.value(*subs)

        return _Fragment()
