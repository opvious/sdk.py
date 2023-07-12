from __future__ import annotations

import dataclasses
from typing import Literal, Mapping, Optional

from ..common import decode_extended_float, Json
from .tensors import is_value, Value


#: Model name
Label = str
"""Model component name"""


@dataclasses.dataclass(frozen=True)
class SourceBinding:
    """Parameter key item binding"""

    dimension_label: Optional[Label]
    """The label of the dimension if the key item corresponds to one"""

    qualifier: Optional[Label]
    """The binding's qualifier, if any"""


def _source_binding_from_json(data: Json) -> SourceBinding:
    return SourceBinding(
        dimension_label=data.get("dimensionLabel"),
        qualifier=data.get("qualifier"),
    )


ObjectiveSense = Literal[
    "MAXIMIZE",
    "MINIMIZE",
]
"""Objective direction"""


@dataclasses.dataclass(frozen=True)
class ObjectiveOutline:
    """Objective metadata"""

    label: Label
    """The objective's unique label"""

    sense: ObjectiveSense
    """Whether this is a maximization (or minimization)"""

    is_quadratic: bool
    """Whether the objective has any quadratic coefficients"""


def _objective_from_json(data: Json) -> ObjectiveOutline:
    return ObjectiveOutline(
        label=data["label"],
        sense="MAXIMIZE" if data["isMaximization"] else "MINIMIZE",
        is_quadratic=data["isQuadratic"],
    )


@dataclasses.dataclass(frozen=True)
class DimensionOutline:
    """Dimension metadata"""

    label: Label
    """The dimension's unique label"""

    is_numeric: bool
    """Whether the dimension contains numeric items"""


def _dimension_from_json(data: Json) -> DimensionOutline:
    return DimensionOutline(label=data["label"], is_numeric=data["isNumeric"])


@dataclasses.dataclass(frozen=True)
class TensorOutline:
    """Parameter or variable metadata"""

    label: Label
    """The tensor's unique label"""

    lower_bound: Optional[Value]
    """The tensor's lower bound if it is statically known"""

    upper_bound: Optional[Value]
    """The tensor's upper bound if it is statically known"""

    is_integral: bool
    """Whether the tensor contains only integer values"""

    bindings: list[SourceBinding]
    """Key bindings"""

    derivation_kind: Optional[str]
    """Derived tensor kind if applicable"""

    @property
    def is_indicator(self) -> bool:
        """Whether the tensor is guaranteed to contain only 0s and 1s"""
        return (
            self.is_integral
            and self.lower_bound == 0
            and self.upper_bound == 1
        )


def _tensor_from_json(data: Json) -> TensorOutline:
    img = data["image"]
    lb = decode_extended_float(img["lowerBound"])
    ub = decode_extended_float(img["upperBound"])
    dv = data.get("derivation")
    return TensorOutline(
        label=data["label"],
        lower_bound=lb if is_value(lb) else None,
        upper_bound=ub if is_value(ub) else None,
        is_integral=img["isIntegral"],
        bindings=[_source_binding_from_json(b) for b in data["bindings"]],
        derivation_kind=dv["kind"] if dv else None,
    )


@dataclasses.dataclass(frozen=True)
class ConstraintOutline:
    """Constraint metadata"""

    label: Label
    """The constraint's unique label"""

    bindings: list[SourceBinding]
    """Quantifier key bindings"""


def _constraint_from_json(data: Json) -> ConstraintOutline:
    return ConstraintOutline(
        label=data["label"],
        bindings=[_source_binding_from_json(b) for b in data["bindings"]],
    )


@dataclasses.dataclass(frozen=True)
class Outline:
    """Model metadata"""

    objectives: Mapping[Label, ObjectiveOutline]
    """Objective metadata, if applicable"""

    dimensions: Mapping[Label, DimensionOutline]
    """Dimension metadata, keyed by dimension label"""

    parameters: Mapping[Label, TensorOutline]
    """Parameter metadata, keyed by parameter label"""

    variables: Mapping[Label, TensorOutline]
    """Variable metadata, keyed by variable label"""

    constraints: Mapping[Label, ConstraintOutline]
    """Constraint metadata, keyed by constraint label"""


def outline_from_json(data: Json) -> Outline:
    return Outline(
        objectives=_map_outlines(_objective_from_json, data["objectives"]),
        dimensions=_map_outlines(_dimension_from_json, data["dimensions"]),
        parameters=_map_outlines(_tensor_from_json, data["parameters"]),
        variables=_map_outlines(_tensor_from_json, data["variables"]),
        constraints=_map_outlines(_constraint_from_json, data["constraints"]),
    )


def _map_outlines(from_json, data):
    return {o["label"]: from_json(o) for o in data}
