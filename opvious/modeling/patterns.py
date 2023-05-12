from typing import Optional, Union

from .images import indicator
from .model import Variable, Constraint, constraint, ExpressionLike


def activation(
    variable: Variable,
    upper_bound: Union[ExpressionLike, bool] = False,
    lower_bound: Union[ExpressionLike, bool] = False,
) -> tuple[Variable, Optional[Constraint], Optional[Constraint]]:
    activation = Variable(variable.quantification(), image=indicator())

    if upper_bound is False:
        activates = None
    else:
        if upper_bound is True:
            upper_bound = variable.image.upper_bound

        @constraint()
        def activates(_model):
            for t in variable.quantification():
                yield variable(*t) <= upper_bound * activation(*t)

    if lower_bound is False:
        deactivates = None
    else:
        if lower_bound is True:
            lower_bound = variable.image.lower_bound

        @constraint()
        def deactivates(_model):
            for t in variable.quantification():
                yield variable(*t) >= lower_bound * activation(*t)

    return activation, activates, deactivates
