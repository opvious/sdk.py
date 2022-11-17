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
import pandas as pd
from typing import Any, Iterable, Mapping, Optional, Union


KeyItem = Union[float, int, str]


Key = Union[tuple[KeyItem, ...], KeyItem]


Value = Union[float, int]


def is_value(arg: Any) -> bool:
    return isinstance(arg, (float, int))


Label = str


DimensionArgument = Iterable[KeyItem]


SparseParameterArgument = Union[
    pd.Series,
    pd.DataFrame,
    Mapping[Key, Value],
]


ParameterArgument = Union[
    Value, SparseParameterArgument, tuple[SparseParameterArgument, Value]
]


# Outlines


@dataclasses.dataclass
class SourceBinding:
    dimension_label: Optional[Label]
    qualifier: Optional[Label]

    @classmethod
    def from_graphql(cls, data: Any) -> SourceBinding:
        return SourceBinding(
            dimension_label=data["dimensionLabel"],
            qualifier=data["qualifier"],
        )


@dataclasses.dataclass
class ObjectiveOutline:
    is_maximization: bool

    @classmethod
    def from_graphql(cls, data: Any) -> ObjectiveOutline:
        return ObjectiveOutline(is_maximization=data["isMaximization"])


@dataclasses.dataclass
class DimensionOutline:
    label: Label
    is_numeric: bool

    @classmethod
    def from_graphql(cls, data: Any) -> DimensionOutline:
        return DimensionOutline(
            label=data["label"], is_numeric=data["isNumeric"]
        )


@dataclasses.dataclass
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
    def from_graphql(cls, data: Any) -> TensorOutline:
        lb = data["lowerBound"]
        ub = data["upperBound"]
        return TensorOutline(
            label=data["label"],
            lower_bound=lb if is_value(lb) else None,
            upper_bound=ub if is_value(ub) else None,
            is_integral=data["isIntegral"],
            bindings=[SourceBinding.from_graphql(b) for b in data["bindings"]],
        )


@dataclasses.dataclass
class ConstraintOutline:
    label: Label
    bindings: list[SourceBinding]

    @classmethod
    def from_graphql(cls, data: Any) -> ConstraintOutline:
        return ConstraintOutline(
            label=data["label"],
            bindings=[SourceBinding.from_graphql(b) for b in data["bindings"]],
        )


@dataclasses.dataclass
class Outline:
    objective: Optional[ObjectiveOutline]
    dimensions: Mapping[Label, DimensionOutline]
    parameters: Mapping[Label, TensorOutline]
    variables: Mapping[Label, TensorOutline]
    constraints: Mapping[Label, ConstraintOutline]

    @classmethod
    def from_graphql(cls, data: Any) -> Outline:
        obj = data["objective"]
        return Outline(
            objective=ObjectiveOutline.from_graphql(obj) if obj else None,
            dimensions=_map_outlines(DimensionOutline, data["dimensions"]),
            parameters=_map_outlines(TensorOutline, data["parameters"]),
            variables=_map_outlines(TensorOutline, data["variables"]),
            constraints=_map_outlines(ConstraintOutline, data["constraints"]),
        )


def _map_outlines(cls, data):
    return {o["label"]: cls.from_graphql(o) for o in data}


# Outcomes


@dataclasses.dataclass
class FailedOutcome:
    status: str
    message: str
    code: Optional[str]
    tags: Any

    @classmethod
    def from_graphql(cls, data: Any) -> FailedOutcome:
        failure = data["failure"]
        return FailedOutcome(
            status=failure["status"],
            message=failure["message"],
            code=failure.get("code"),
            tags=failure.get("tags"),
        )


@dataclasses.dataclass
class FeasibleOutcome:
    is_optimal: bool
    objective_value: Value
    relative_gap: Optional[Value]

    @classmethod
    def from_graphql(cls, data: Any) -> FeasibleOutcome:
        return FeasibleOutcome(
            is_optimal=data["isOptimal"],
            objective_value=data["objectiveValue"],
            relative_gap=data.get("relativeGap"),
        )


@dataclasses.dataclass
class InfeasibleOutcome:
    pass


@dataclasses.dataclass
class UnboundedOutcome:
    pass


Outcome = Union[
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
]


# Attempt


@dataclasses.dataclass
class Inputs:
    formulation_name: str
    tag_name: str
    outline: Outline
    dimensions: list[Any]
    parameters: list[Any]


@dataclasses.dataclass
class Attempt:
    uuid: str
    outline: Outline
    url: str


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
