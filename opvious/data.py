"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
import pandas as pd
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

from .common import strip_nones


KeyItem = Union[float, int, str]


Key = Union[Tuple[KeyItem, ...], KeyItem]


Value = Union[float, int]


def is_value(arg: Any) -> bool:
    return isinstance(arg, (float, int))


Label = str


DimensionArgument = Iterable[KeyItem]


# Tensors


SparseTensorArgument = Union[
    pd.Series,
    Mapping[Key, Value],
    pd.DataFrame,  # For indicator parameters
    Iterable[Key],  # For indicator parameters
]


TensorArgument = Union[
    Value, SparseTensorArgument, Tuple[SparseTensorArgument, Value]
]


@dataclasses.dataclass
class Tensor:
    entries: List[Any]
    default_value: float = 0

    @classmethod
    def from_argument(cls, arg: TensorArgument, is_indicator: bool = False):
        if isinstance(arg, tuple):
            data, default_value = arg
        else:
            data = arg
            default_value = 0
        if (
            is_indicator
            and isinstance(data, pd.Series)
            and not pd.api.types.is_numeric_dtype(data)
        ):
            data = data.reset_index()
        if is_indicator and isinstance(data, pd.DataFrame):
            entries = [
                {"key": key, "value": 1}
                for key in data.itertuples(index=False, name=None)
            ]
        elif is_indicator and not hasattr(data, "items"):
            entries = [{"key": key, "value": 1} for key in data]
        else:
            if is_value(data):
                entries = [{"key": (), "value": data}]
            else:
                entries = [
                    {"key": _keyify(key), "value": value}
                    for key, value in data.items()
                ]
        return Tensor(entries, default_value)


def _keyify(key):
    return tuple(key) if isinstance(key, (list, tuple)) else (key,)


# Outlines


@dataclasses.dataclass
class SourceBinding:
    dimension_label: Optional[Label]
    qualifier: Optional[Label]

    @classmethod
    def from_json(cls, data: Any) -> SourceBinding:
        return SourceBinding(
            dimension_label=data.get("dimensionLabel"),
            qualifier=data.get("qualifier"),
        )


@dataclasses.dataclass
class ObjectiveOutline:
    is_maximization: bool

    @classmethod
    def from_json(cls, data: Any) -> ObjectiveOutline:
        return ObjectiveOutline(is_maximization=data["isMaximization"])


@dataclasses.dataclass
class DimensionOutline:
    label: Label
    is_numeric: bool

    @classmethod
    def from_json(cls, data: Any) -> DimensionOutline:
        return DimensionOutline(
            label=data["label"], is_numeric=data["isNumeric"]
        )


@dataclasses.dataclass
class TensorOutline:
    label: Label
    lower_bound: Optional[Value]
    upper_bound: Optional[Value]
    is_integral: bool
    bindings: List[SourceBinding]

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


@dataclasses.dataclass
class ConstraintOutline:
    label: Label
    bindings: List[SourceBinding]

    @classmethod
    def from_json(cls, data: Any) -> ConstraintOutline:
        return ConstraintOutline(
            label=data["label"],
            bindings=[SourceBinding.from_json(b) for b in data["bindings"]],
        )


@dataclasses.dataclass
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


# Outcomes


@dataclasses.dataclass
class CancelledOutcome:
    reached_at: datetime


@dataclasses.dataclass
class FailedOutcome:
    reached_at: datetime
    status: str
    message: str
    code: Optional[str]
    tags: Any

    @classmethod
    def from_graphql(cls, reached_at: datetime, data: Any) -> FailedOutcome:
        failure = data["failure"]
        return FailedOutcome(
            reached_at=reached_at,
            status=failure["status"],
            message=failure["message"],
            code=failure.get("code"),
            tags=failure.get("tags"),
        )


@dataclasses.dataclass
class FeasibleOutcome:
    reached_at: datetime
    is_optimal: bool
    objective_value: Optional[Value]
    relative_gap: Optional[Value]

    @classmethod
    def from_graphql(cls, reached_at: datetime, data: Any) -> FeasibleOutcome:
        return FeasibleOutcome(
            reached_at=reached_at,
            is_optimal=data["isOptimal"],
            objective_value=data.get("objectiveValue"),
            relative_gap=data.get("relativeGap"),
        )


@dataclasses.dataclass
class InfeasibleOutcome:
    reached_at: datetime


@dataclasses.dataclass
class UnboundedOutcome:
    reached_at: datetime


Outcome = Union[
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
]


# Summaries


@dataclasses.dataclass
class Summary:
    column_count: int
    row_count: int
    weight_count: int

    def density(self) -> float:
        denom = self.column_count * self.row_count
        return self.weight_count / denom if denom > 0 else 1

    @classmethod
    def from_json(cls, data) -> "Summary":
        column_count = 0
        for item in data["variables"]:
            column_count += item["columnCount"]
        row_count = 0
        weight_count = 0
        for item in data["constraints"]:
            row_count += item["rowCount"]
            weight_count += item["weightProfile"]["count"]
        return Summary(
            column_count=column_count,
            row_count=row_count,
            weight_count=weight_count,
        )


# Solve data


@dataclasses.dataclass
class InputData:
    outline: Outline
    raw_parameters: List[Any]
    raw_dimensions: Optional[List[Any]]

    def parameter(self, label: Label) -> pd.Series:
        for param in self.raw_parameters:
            if param["label"] == label:
                entries = param["entries"]
                outline = self.outline.parameters[label]
                return pd.Series(
                    data=(e["value"] for e in entries),
                    index=_entry_index(entries, outline.bindings),
                )
        raise Exception(f"Unknown parameter: {label}")

    def dimension(self, label: Label) -> pd.Series:
        for dim in self.raw_dimensions or []:
            if dim["label"] == label:
                return pd.Index(dim["items"])
        raise Exception(f"Unknown dimension: {label}")


@dataclasses.dataclass
class OutputData:
    outline: Outline
    raw_variables: List[Any]
    raw_constraints: List[Any]

    @classmethod
    def from_json(cls, data, outline):
        return OutputData(
            outline=outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )

    def variable(self, label: Label) -> pd.DataFrame:
        for res in self.raw_variables:
            if res["label"] == label:
                entries = res["entries"]
                outline = self.outline.variables[label]
                df = pd.DataFrame(
                    data=(
                        {"value": e["value"], "dual_value": e.get("dualValue")}
                        for e in entries
                    ),
                    index=_entry_index(entries, outline.bindings),
                )
                return df.dropna(axis=1, how="all").fillna(0)
        raise Exception(f"Unknown variable {label}")

    def constraint(self, label: Label) -> pd.DataFrame:
        for res in self.raw_constraints:
            if res["label"] == label:
                entries = res["entries"]
                outline = self.outline.constraints[label]
                df = pd.DataFrame(
                    data=(
                        {"slack": e["value"], "dual_value": e.get("dualValue")}
                        for e in entries
                    ),
                    index=_entry_index(entries, outline.bindings),
                )
                return df.dropna(axis=1, how="all").fillna(0)
        raise Exception(f"Unknown constraint {label}")


def _entry_index(entries, bindings):
    if len(bindings) == 1:
        binding = bindings[0]
        return pd.Index(
            data=[e["key"][0] for e in entries],
            name=binding.qualifier or binding.dimension_label,
        )
    return pd.MultiIndex.from_tuples(
        tuples=[tuple(e["key"]) for e in entries],
        names=[b.qualifier or b.dimension_label for b in bindings],
    )


@dataclasses.dataclass
class Inputs:
    """Solve inputs"""

    formulation_name: str
    tag_name: str
    data: InputData = dataclasses.field(repr=False)


@dataclasses.dataclass
class Outputs:
    """Solve outputs"""

    status: str
    outcome: Outcome
    summary: Summary
    data: Optional[OutputData] = dataclasses.field(default=None, repr=False)


# Attempt options

Penalty = str


_DEFAULT_PENALTY = "TOTAL_DEVIATION"


@dataclasses.dataclass
class Relaxation:
    penalty: Penalty
    objective_weight: Optional[float] = None
    constraints: Optional[List[ConstraintRelaxation]] = None

    @classmethod
    def from_constraint_labels(cls, labels: List[Label]) -> Relaxation:
        return Relaxation(
            penalty=_DEFAULT_PENALTY,
            constraints=[ConstraintRelaxation(label=n) for n in labels],
        )

    def to_json(self):
        return strip_nones(
            {
                "penalty": self.penalty,
                "objectiveWeight": self.objective_weight,
                "constraints": None
                if self.constraints is None
                else [c.to_json() for c in self.constraints],
            }
        )


@dataclasses.dataclass
class ConstraintRelaxation:
    label: Label
    penalty: Optional[str] = None
    cost: Optional[float] = None
    bound: Optional[float] = None

    def to_json(self):
        return strip_nones(
            {
                "label": self.label,
                "penalty": self.penalty,
                "deficitCost": self.cost,
                "surplusCost": self.cost,
                "deficitBound": None if self.bound is None else -self.bound,
                "surplusBound": self.bound,
            }
        )


@dataclasses.dataclass
class Attempt:
    uuid: str
    started_at: datetime
    outline: Outline
    url: str

    @classmethod
    def from_graphql(cls, data: Any, outline: Outline, url: str):
        return Attempt(
            uuid=data["uuid"],
            started_at=datetime.fromisoformat(data["startedAt"]),
            outline=outline,
            url=url,
        )


@dataclasses.dataclass
class Notification:
    dequeued: bool
    relative_gap: Optional[Value]
    lp_iteration_count: Optional[int]
    cut_count: Optional[int]

    @classmethod
    def from_graphql(cls, dequeued: bool, data: Any = None):
        return Notification(
            dequeued=dequeued,
            relative_gap=data["relativeGap"] if data else None,
            lp_iteration_count=data["lpIterationCount"] if data else None,
            cut_count=data["cutCount"] if data else None,
        )
