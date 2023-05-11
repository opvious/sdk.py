from typing import Optional, Union

from .images import indicator
from .model import Variable, Constraint, constraint, Expression


def activation(
    variable: Variable,
    upper_bound: Union[float, Expression, bool] = False,
    lower_bound: Union[float, Expression, bool] = False,
) -> tuple[Variable, Optional[Constraint], Optional[Constraint]]:
    activation = Variable(variable.quantification(), image=indicator())

    if upper_bound is False:
        activates = None
    else:
        # TODO: Infer bound when `True`
        @constraint()
        def activates(_model):
            for t in variable.quantification():
                yield variable(*t) <= upper_bound * activation(*t)

    if lower_bound is False:
        deactivates = None
    else:
        # TODO: Infer bound when `True`
        @constraint()
        def deactivates(_model):
            for t in variable.quantification():
                yield variable(*t) >= lower_bound * activation(*t)

    return activation, activates, deactivates
