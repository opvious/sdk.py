from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from .tensors import decode_extended_float, is_value, Value


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


def _source_binding_from_json(data: Any) -> SourceBinding:
    return SourceBinding(
        dimension_label=data.get("dimensionLabel"),
        qualifier=data.get("qualifier"),
    )


@dataclasses.dataclass(frozen=True)
class ObjectiveOutline:
    """Objective metadata"""

    is_maximization: bool
    """Whether this is a maximization (or minimization)"""

    is_quadratic: bool
    """Whether the objective has any quadratic coefficients"""


def _objective_from_json(data: Any) -> ObjectiveOutline:
    return ObjectiveOutline(
        is_maximization=data["isMaximization"],
        is_quadratic=data["isQuadratic"],
    )


@dataclasses.dataclass(frozen=True)
class DimensionOutline:
    """Dimension metadata"""

    label: Label
    """The dimension's unique label"""

    is_numeric: bool
    """Whether the dimension contains numeric items"""


def _dimension_from_json(data: Any) -> DimensionOutline:
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

    @property
    def is_indicator(self) -> bool:
        """Whether the tensor is guaranteed to contain only 0s and 1s"""
        return (
            self.is_integral
            and self.lower_bound == 0
            and self.upper_bound == 1
        )


def _tensor_from_json(data: Any) -> TensorOutline:
    lb = decode_extended_float(data["lowerBound"])
    ub = decode_extended_float(data["upperBound"])
    return TensorOutline(
        label=data["label"],
        lower_bound=lb if is_value(lb) else None,
        upper_bound=ub if is_value(ub) else None,
        is_integral=data["isIntegral"],
        bindings=[_source_binding_from_json(b) for b in data["bindings"]],
    )


@dataclasses.dataclass(frozen=True)
class ConstraintOutline:
    """Constraint metadata"""

    label: Label
    """The constraint's unique label"""

    bindings: list[SourceBinding]
    """Quantifier key bindings"""


def _constraint_from_json(data: Any) -> ConstraintOutline:
    return ConstraintOutline(
        label=data["label"],
        bindings=[_source_binding_from_json(b) for b in data["bindings"]],
    )


@dataclasses.dataclass(frozen=True)
class Outline:
    """Model metadata"""

    objective: Optional[ObjectiveOutline]
    """Objective metadata, if applicable"""

    dimensions: Mapping[Label, DimensionOutline]
    """Dimension metadata, keyed by dimension label"""

    parameters: Mapping[Label, TensorOutline]
    """Parameter metadata, keyed by parameter label"""

    variables: Mapping[Label, TensorOutline]
    """Variable metadata, keyed by variable label"""

    constraints: Mapping[Label, ConstraintOutline]
    """Constraint metadata, keyed by constraint label"""


def outline_from_json(data: Any) -> Outline:
    obj = data.get("objective")
    return Outline(
        objective=_objective_from_json(obj) if obj else None,
        dimensions=_map_outlines(_dimension_from_json, data["dimensions"]),
        parameters=_map_outlines(_tensor_from_json, data["parameters"]),
        variables=_map_outlines(_tensor_from_json, data["variables"]),
        constraints=_map_outlines(_constraint_from_json, data["constraints"]),
    )


def _map_outlines(from_json, data):
    return {o["label"]: from_json(o) for o in data}
