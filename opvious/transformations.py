"""Built-in transformations

This module exports all available :class:`~opvious.Transformation` instances.
As a convenience it is also exported by the `opvious` module.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

from .common import Json, json_dict
from .data.outlines import Label, Outline
from .data.solves import Target


class TransformationContext:
    def __init__(self):
        self._json = []

    def add(self, kind: str, **kwargs) -> None:
        self.add_json(json_dict(kind=kind, **kwargs))

    def add_json(self, data: Json) -> None:
        self._json.append(data)

    def get_json(self) -> list[Json]:
        return self._json

    async def fetch_outline(self) -> Outline:
        raise NotImplementedError()


class Transformation:
    """Base transformation class

    You should not need to interact with this class directly, instead use one
    of the available :ref:`transformation subclasses <Base transformations>`.
    """

    async def register(self, context: TransformationContext) -> None:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class PinVariables(Transformation):
    """A transformation which pins one or more variables to input values

    Each pinned variable will have an associated derived parameter labeled
    `$label_pin` with the same domain and image as the original variable. For
    example a non-negative `production` variable would have a non-negative
    parameter labeled `production_pin`.

    A constraint (labeled `$label_isPinned`) is also automatically added which
    enforces equality between the variable and any inputs passed in to the
    parameter. Values for keys which do not have a pin remain free, so it is
    possible to do partial pinning. For example, assuming a monthly variable
    labeled `production`, the following code would only pin production in
    January (to 100):

    .. code-block:: python

        transformations = [opvious.transformations.PinVariables("production")]
        parameters = {
            "production_pin": {"january": 100},
            # ...
        }

    Finally, note that this constraint can also be transformed similar to any
    other constraint. For example it can sometimes be useful to relax it via
    :class:`.RelaxConstraints`.
    """

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the variables to pin

    If empty, all variables will be pinned.
    """

    async def register(self, context: TransformationContext) -> None:
        labels = self.labels
        if not labels:
            outline = await context.fetch_outline()
            labels = list(outline.variables)
        for label in labels:
            context.add("pinVariable", label=label)


RelaxationPenalty = Literal[
    "TOTAL_DEVIATION",
    "MAX_DEVIATION",
    "DEVIATION_CARDINALITY",
]


@dataclasses.dataclass(frozen=True)
class RelaxConstraints(Transformation):
    """A transformation which relaxes one or more constraints

    Each relaxed constraint will be omitted from the formulation and replaced
    by a slack variable and an objective minimizing this slack (two of each for
    equality constraints). The derived variables have the same domain as the
    relaxed constraint, are always non-negative, and are labeled as follows:

    + `$label_deficit` for deficit slack (applicable for greater than and
      equality constraints).
    + `$label_surplus` for surplus slack (applicable for less than and equality
      constraints).

    For example an equality constraint labeled `isBalanced` would be
    transformed into two variables labeled `isBalanced_deficit` and
    `isBalanced_surplus`. A greater than constraint labeled `demandIsMet` would
    be relaxed into a single variable labeled `demandIsMet_deficit`.

    Each deficit (resp. surplus) variable has a corresponding objective labeled
    `$label_minimizeDeficit` (resp. `$label_minimizeSurplus`). Refer to the
    `penalty` parameter for details on how slack is penalized.

    Finally, since relaxed formulations will almost always have multiple
    objectives, you may also need to specific a :class:`opvious.SolveStrategy`.
    A common pattern is to omit any existing objectives and minimize the
    aggregate slack violation (see :ref:`Detecting infeasibilities`).
    """

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the constraints to relax

    If empty, all constraints will be relaxed.
    """

    penalty: RelaxationPenalty = "TOTAL_DEVIATION"
    """Slack penalization mode

    + `TOTAL_DEVIATION`: Cost proportional to the total sum of the (absolute
      value of) slack for the constraint.
    + `MAX_DEVIATION`: Cost proportional to the maximum deviation for the
      constraint.
    + `DEVIATION_CARDINALITY`: Cost proportional to the number of rows with
      non-zero deviation. This penalty requires the relaxation to be capped.

    The default is `TOTAL_DEVIATION`.
    """

    is_capped: bool = False
    """Whether slack is capped

    Setting this to true will create an additional parameter for each slack
    variable labeled `$label_deficitCap` (resp. `$label_surplusCap`) for
    deficit (resp. surplus). This parameter has the same domain as the variable
    and is non-negative.

    This option is required when using the `DEVIATION_CARDINALITY` penalty.
    """

    async def register(self, context: TransformationContext) -> None:
        labels = self.labels
        if not labels:
            outline = await context.fetch_outline()
            labels = list(outline.constraints)
        for label in labels:
            context.add(
                "relaxConstraint",
                label=label,
                penalty=self.penalty,
                is_capped=self.is_capped,
            )


@dataclasses.dataclass(frozen=True)
class DensifyVariables(Transformation):
    """A transformation which updates one or more variables to be continuous"""

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the variables to densify

    If empty, all integral variables will be densified.
    """

    async def register(self, _context: TransformationContext) -> None:
        raise NotImplementedError()  # TODO


@dataclasses.dataclass(frozen=True)
class OmitConstraints(Transformation):
    """A transformation which drops one or more constraints

    Any parameters or variables which are not referenced in any remaining
    constraint or objective will automatically be dropped.
    """

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the constraints to drop

    If empty, all constraints will be dropped.
    """

    async def register(self, context: TransformationContext) -> None:
        labels = self.labels
        if not labels:
            outline = await context.fetch_outline()
            labels = list(outline.constraints)
        for label in labels:
            context.add("omitConstraint", label=label)


@dataclasses.dataclass(frozen=True)
class OmitObjectives(Transformation):
    """A transformation which drops one or more objectives

    Similar to :class:`.OmitConstraints`, any parameters or variables which are
    not referenced in any remaining constraint or objective will automatically
    be dropped.
    """

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the objectives to drop

    If empty, all objectives will be dropped.
    """

    async def register(self, context: TransformationContext) -> None:
        labels = self.labels
        if not labels:
            outline = await context.fetch_outline()
            labels = list(outline.objectives)
        for label in labels:
            context.add("omitObjective", label=label)


@dataclasses.dataclass(frozen=True)
class ConstrainObjective(Transformation):
    """A transformation which bounds the value of a objective

    This can be used for example to guarantee a minimum objective levels when
    multiple objectives are involved or to implement custom multi-objective
    strategies (see :ref:`Weighted distance multi-objective optimization`).
    """

    target: Target
    """The label of the objective to constrain"""

    min_value: float = -math.inf
    """The objective's minimum allowed value"""

    max_value: float = math.inf
    """The objective's maximum allowed value"""

    async def register(self, _context: TransformationContext) -> None:
        raise NotImplementedError()  # TODO
