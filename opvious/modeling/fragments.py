from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from ..common import Label, untuple
from .ast import cross, Quantifiable
from .identifiers import Name
from .images import Image
from .model import (
    alias,
    Parameter,
    Variable,
    constraint,
    Expression,
    ExpressionLike,
    ModelFragment,
)
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
        upper_bound: Union[ExpressionLike, bool] = False,
        lower_bound: Union[ExpressionLike, bool] = False,
    ) -> ActivationIndicator:
        """Returns a variable activation fragment"""

        class _Fragment(ActivationIndicator):
            is_active = Variable(
                variable.quantification(),
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
                    bound = variable.image.upper_bound
                for t in variable.quantification():
                    yield variable(*t) >= bound * self.is_active(*t)

        return _Fragment()


class MaskedSubset(ModelFragment):
    """Quantifiable subset"""

    def __iter__(self) -> Iterable[Any]:
        raise NotImplementedError()

    @classmethod
    def fragment(
        cls,
        *quantifiables: Quantifiable,
        parameter_label: Optional[Label] = None,
        alias_name: Optional[Name] = None,
    ) -> MaskedSubset:
        """Returns a quantifiable subset fragment"""

        class _Fragment(MaskedSubset):
            mask = Parameter(
                quantifiables,
                image=Image.indicator(),
                label=parameter_label,
            )

            @property
            @alias(alias_name)
            def masked(self) -> Quantified:
                for t in cross(quantifiables):
                    if self.mask(*t):
                        yield untuple(t)

            def __iter__(self):
                return self.masked

        return _Fragment()
