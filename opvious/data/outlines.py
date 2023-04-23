from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from .tensors import is_value, Value


#: Model name
Label = str
"""Model component name"""


@dataclasses.dataclass(frozen=True)
class SourceBinding:
    dimension_label: Optional[Label]
    qualifier: Optional[Label]

    @classmethod
    def from_json(cls, data: Any) -> SourceBinding:
        return SourceBinding(
            dimension_label=data.get("dimensionLabel"),
            qualifier=data.get("qualifier"),
        )


@dataclasses.dataclass(frozen=True)
class ObjectiveOutline:
    is_maximization: bool

    @classmethod
    def from_json(cls, data: Any) -> ObjectiveOutline:
        return ObjectiveOutline(is_maximization=data["isMaximization"])


@dataclasses.dataclass(frozen=True)
class DimensionOutline:
    label: Label
    is_numeric: bool

    @classmethod
    def from_json(cls, data: Any) -> DimensionOutline:
        return DimensionOutline(
            label=data["label"], is_numeric=data["isNumeric"]
        )


@dataclasses.dataclass(frozen=True)
class TensorOutline:
    label: Label
    lower_bound: Optional[Value]
    upper_bound: Optional[Value]
    is_integral: bool
    bindings: list[SourceBinding]

    def is_indicator(self) -> bool:
        return (
            self.is_integral
            and self.lower_bound == 0
            and self.upper_bound == 1
        )

    @classmethod
    def from_json(cls, data: Any) -> TensorOutline:
        lb = data["lowerBound"]
        ub = data["upperBound"]
        return TensorOutline(
            label=data["label"],
            lower_bound=lb if is_value(lb) else None,
            upper_bound=ub if is_value(ub) else None,
            is_integral=data["isIntegral"],
            bindings=[SourceBinding.from_json(b) for b in data["bindings"]],
        )


@dataclasses.dataclass(frozen=True)
class ConstraintOutline:
    label: Label
    bindings: list[SourceBinding]

    @classmethod
    def from_json(cls, data: Any) -> ConstraintOutline:
        return ConstraintOutline(
            label=data["label"],
            bindings=[SourceBinding.from_json(b) for b in data["bindings"]],
        )


@dataclasses.dataclass(frozen=True)
class Outline:
    objective: Optional[ObjectiveOutline]
    dimensions: Mapping[Label, DimensionOutline]
    parameters: Mapping[Label, TensorOutline]
    variables: Mapping[Label, TensorOutline]
    constraints: Mapping[Label, ConstraintOutline]

    @classmethod
    def from_json(cls, data: Any) -> Outline:
        obj = data.get("objective")
        return Outline(
            objective=ObjectiveOutline.from_json(obj) if obj else None,
            dimensions=_map_outlines(DimensionOutline, data["dimensions"]),
            parameters=_map_outlines(TensorOutline, data["parameters"]),
            variables=_map_outlines(TensorOutline, data["variables"]),
            constraints=_map_outlines(ConstraintOutline, data["constraints"]),
        )


def _map_outlines(cls, data):
    return {o["label"]: cls.from_json(o) for o in data}
