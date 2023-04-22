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
from typing import Any, cast, Optional

from ..common import strip_nones
from .outcomes import (
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
    Outcome,
)
from .outlines import Label, Outline, SourceBinding
from .tensors import decode_extended_float


@dataclasses.dataclass(frozen=True)
class SolveSummary:
    column_count: int
    row_count: int
    weight_count: int

    def density(self) -> float:
        denom = self.column_count * self.row_count
        return self.weight_count / denom if denom > 0 else 1

    @classmethod
    def from_json(cls, data) -> "SolveSummary":
        column_count = 0
        for item in data["variables"]:
            column_count += item["columnCount"]
        row_count = 0
        weight_count = 0
        for item in data["constraints"]:
            row_count += item["rowCount"]
            weight_count += item["weightProfile"]["count"]
        return SolveSummary(
            column_count=column_count,
            row_count=row_count,
            weight_count=weight_count,
        )


def _entry_index(entries, bindings):
    if not bindings:
        return None
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


@dataclasses.dataclass(frozen=True)
class SolveInputs:
    outline: Outline
    raw_parameters: list[Any]
    raw_dimensions: Optional[list[Any]]

    def parameter(self, label: Label) -> pd.Series:
        for param in self.raw_parameters:
            if param["label"] == label:
                entries = param["entries"]
                outline = self.outline.parameters[label]
                return pd.Series(
                    data=(decode_extended_float(e["value"]) for e in entries),
                    index=_entry_index(entries, outline.bindings),
                )
        raise Exception(f"Unknown parameter: {label}")

    def dimension(self, label: Label) -> pd.Index:
        for dim in self.raw_dimensions or []:
            if dim["label"] == label:
                return pd.Index(dim["items"])
        raise Exception(f"Unknown dimension: {label}")


@dataclasses.dataclass(frozen=True)
class SolveOutputs:
    outline: Outline
    raw_variables: list[Any]
    raw_constraints: list[Any]

    @classmethod
    def from_json(cls, data, outline):
        return SolveOutputs(
            outline=outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )

    def _variable_bindings(self, label: Label) -> list[SourceBinding]:
        prefix = label.split("_", 1)[0]
        if prefix == label:
            return self.outline.variables[prefix].bindings
        return self.outline.constraints[prefix].bindings

    def variable(self, label: Label) -> pd.DataFrame:
        for res in self.raw_variables:
            if res["label"] == label:
                entries = res["entries"]
                bindings = self._variable_bindings(label)
                df = pd.DataFrame(
                    data=(
                        {
                            "value": decode_extended_float(e["value"]),
                            "dual_value": e.get("dualValue"),
                        }
                        for e in entries
                    ),
                    index=_entry_index(entries, bindings),
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
                        {
                            "slack": e["value"],
                            "dual_value": e.get("dualValue"),
                        }
                        for e in entries
                    ),
                    index=_entry_index(entries, outline.bindings),
                )
                return df.dropna(axis=1, how="all").fillna(0)
        raise Exception(f"Unknown constraint {label}")


@dataclasses.dataclass(frozen=True)
class SolveResponse:
    status: str
    outcome: Outcome
    summary: SolveSummary
    outputs: Optional[SolveOutputs] = dataclasses.field(
        default=None, repr=False
    )

    @classmethod
    def from_json(
        cls,
        outline: Outline,
        response_json: Any,
        summary: Optional[SolveSummary] = None,
    ) -> "SolveResponse":
        outcome_json = response_json["outcome"]
        status = outcome_json["status"]
        if status == "INFEASIBLE":
            outcome = cast(Outcome, InfeasibleOutcome())
        elif status == "UNBOUNDED":
            outcome = UnboundedOutcome()
        else:
            outcome = FeasibleOutcome(
                is_optimal=status == "OPTIMAL",
                objective_value=outcome_json.get("objectiveValue"),
                relative_gap=outcome_json.get("relativeGap"),
            )
        outputs = None
        if isinstance(outcome, FeasibleOutcome):
            outputs = SolveOutputs.from_json(
                data=response_json["outputs"],
                outline=outline,
            )
        return SolveResponse(
            status=status,
            outcome=outcome,
            summary=summary
            or SolveSummary.from_json(response_json["summary"]),
            outputs=outputs,
        )


RelaxationPenalty = str


_DEFAULT_PENALTY = "TOTAL_DEVIATION"


@dataclasses.dataclass(frozen=True)
class Relaxation:
    penalty: RelaxationPenalty = _DEFAULT_PENALTY
    objective_weight: Optional[float] = None
    constraints: Optional[list[ConstraintRelaxation]] = None

    @classmethod
    def from_constraint_labels(
        cls,
        labels: list[Label],
        penalty: RelaxationPenalty = _DEFAULT_PENALTY,
        objective_weight: Optional[float] = None,
    ) -> Relaxation:
        """Relaxes all input constraints using a common penalty."""
        return Relaxation(
            penalty=penalty,
            objective_weight=objective_weight,
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


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class SolveOptions:
    relative_gap_threshold: Optional[float] = None
    absolute_gap_threshold: Optional[float] = None
    zero_value_threshold: Optional[float] = None
    infinity_value_threshold: Optional[float] = None
    free_bound_threshold: Optional[float] = None
    timeout_millis: Optional[float] = None


def solve_options_to_json(
    options: Optional[SolveOptions] = None,
    relaxation: Optional[Relaxation] = None,
):
    if not options:
        options = SolveOptions()
    return strip_nones(
        {
            "absoluteGapThreshold": options.absolute_gap_threshold,
            "relativeGapThreshold": options.relative_gap_threshold,
            "timeoutMillis": options.timeout_millis,
            "zeroValueThreshold": options.zero_value_threshold,
            "infinityValueThreshold": options.infinity_value_threshold,
            "freeBoundThreshold": options.free_bound_threshold,
            "relaxation": relaxation.to_json() if relaxation else None,
        }
    )
