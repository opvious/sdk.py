from __future__ import annotations

import dataclasses
import math
from typing import Literal

from .common import Json, json_dict
from .data.outlines import Label, Outline
from .data.solves import WeightedSumTarget


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
    async def register(self, context: TransformationContext) -> None:
        raise NotImplementedError()


RelaxationPenalty = Literal[
    "TOTAL_DEVIATION",
    "MAX_DEVIATION",
    "DEVIATION_CARDINALITY",
]


@dataclasses.dataclass(frozen=True)
class RelaxConstraints(Transformation):
    """Relaxes one or more constraints"""

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the constraints to relax

    If empty, all constraints will be relaxed.
    """

    penalty: RelaxationPenalty = "TOTAL_DEVIATION"
    """Slack penalization mode"""

    is_capped: bool = False
    """Whether slack bounds will be provided"""

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
class PinVariables(Transformation):
    """Pin variable(s) to select values"""

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the variables to pin

    If empty, all variables will expect pins.
    """

    async def register(self, context: TransformationContext) -> None:
        labels = self.labels
        if not labels:
            outline = await context.fetch_outline()
            labels = list(outline.variables)
        for label in labels:
            context.add("pinVariable", label=label)


@dataclasses.dataclass(frozen=True)
class OmitObjectives(Transformation):
    """Drops objective(s)"""

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
class OmitConstraints(Transformation):
    """Drops constraint(s)"""

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
class ConstrainObjective(Transformation):
    """
    Can be used to implement weighted distance multi-objective strategy:

        transformations=[
            opvious.constrain_objective("foo", min_value=5),
            opvious.constrain_objective("bar", min_value=10),
            opvious.omit_objectives("[foo", "bar"]),
            opvious.relax_constraints(["foo_minValue", "bar_minValue"]),
        ],
        strategy=opvious.Strategy.sum(),
    """

    target: WeightedSumTarget
    min_value: float = -math.inf
    max_value: float = math.inf

    async def register(self, _context: TransformationContext) -> None:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class DensifyVariables(Transformation):
    """Transforms variables from integral to continuous"""

    labels: list[Label] = dataclasses.field(default_factory=lambda: [])
    """The labels of the variables to densify

    If empty, all integral variables will be densified.
    """

    async def register(self, _context: TransformationContext) -> None:
        raise NotImplementedError()
