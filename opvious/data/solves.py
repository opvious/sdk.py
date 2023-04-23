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
    SolveStatus,
)
from .outlines import Label, Outline, SourceBinding
from .tensors import decode_extended_float


@dataclasses.dataclass(frozen=True)
class SolveSummary:
    """Solve summary statistics"""

    column_count: int
    """Total number of variable columns in the reified problem"""

    row_count: int
    """Total number of constraint rows in the reified problem"""

    weight_count: int
    """Total number of non-zero weights in the reified problem"""

    @property
    def density(self) -> float:
        """Fraction of non-zero weights in the constraint matrix"""
        denom = self.column_count * self.row_count
        return self.weight_count / denom if denom > 0 else 1


def solve_summary_from_json(data) -> SolveSummary:
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
    """Solve input data"""

    outline: Outline
    """Target model metadata"""

    raw_parameters: list[Any]
    """All parameters in raw format"""

    raw_dimensions: Optional[list[Any]]
    """All dimensions in raw format"""

    def parameter(self, label: Label) -> pd.Series:
        """Returns the parameter for a given label as a pandas Series"""
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
        """Returns the dimension for a given label as a pandas Index"""
        for dim in self.raw_dimensions or []:
            if dim["label"] == label:
                return pd.Index(dim["items"])
        raise Exception(f"Unknown dimension: {label}")


@dataclasses.dataclass(frozen=True)
class SolveOutputs:
    """Successful solve output data"""

    outline: Outline
    """Solved model metadata"""

    raw_variables: list[Any]
    """All variables in raw format"""

    raw_constraints: list[Any]
    """All constraints in raw format"""

    def _variable_bindings(self, label: Label) -> list[SourceBinding]:
        prefix = label.split("_", 1)[0]
        if prefix == label:
            return self.outline.variables[prefix].bindings
        return self.outline.constraints[prefix].bindings

    def variable(self, label: Label) -> pd.DataFrame:
        """Returns variable results for a given label

        The returned dataframe has two columns: `value` and `dual_value`.
        """
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
        """Returns constraint results for a given label.

        The returned dataframe has two columsn: `slack` and `dual_value`.
        """
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


def _outputs_from_json(data, outline) -> SolveOutputs:
    return SolveOutputs(
        outline=outline,
        raw_variables=data["variables"],
        raw_constraints=data["constraints"],
    )


@dataclasses.dataclass(frozen=True)
class SolveResponse:
    """Solver response"""

    status: SolveStatus
    """Status string"""

    summary: SolveSummary
    """Solve summary statistics"""

    outcome: Outcome
    """Solution metadata"""

    outputs: Optional[SolveOutputs] = dataclasses.field(
        default=None, repr=False
    )
    """Solution data, present iff a solution was found"""


def solve_response_from_json(
    outline: Outline,
    response_json: Any,
    summary: Optional[SolveSummary] = None,
) -> SolveResponse:
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
        outputs = _outputs_from_json(
            data=response_json["outputs"],
            outline=outline,
        )
    return SolveResponse(
        status=status,
        outcome=outcome,
        summary=summary or solve_summary_from_json(response_json["summary"]),
        outputs=outputs,
    )


RelaxationPenalty = str


_DEFAULT_PENALTY = "TOTAL_DEVIATION"


@dataclasses.dataclass(frozen=True)
class Relaxation:
    """Problem relaxation configuration"""

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
        """Relaxes all input constraints using a common penalty"""
        return Relaxation(
            penalty=penalty,
            objective_weight=objective_weight,
            constraints=[ConstraintRelaxation(label=n) for n in labels],
        )


def _relaxation_to_json(r: Relaxation) -> Any:
    return strip_nones(
        {
            "penalty": r.penalty,
            "objectiveWeight": r.objective_weight,
            "constraints": None
            if r.constraints is None
            else [_constraint_relaxation_to_json(c) for c in r.constraints],
        }
    )


@dataclasses.dataclass(frozen=True)
class ConstraintRelaxation:
    """Constraint relaxation configuration"""

    label: Label
    penalty: Optional[str] = None
    cost: Optional[float] = None
    bound: Optional[float] = None


def _constraint_relaxation_to_json(c):
    return strip_nones(
        {
            "label": c.label,
            "penalty": c.penalty,
            "deficitCost": c.cost,
            "surplusCost": c.cost,
            "deficitBound": None if c.bound is None else -c.bound,
            "surplusBound": c.bound,
        }
    )


@dataclasses.dataclass(frozen=True)
class SolveOptions:
    """Solving options"""

    relative_gap_threshold: Optional[float] = None
    """Relative gap threshold below which a solution is considered optimal

    For example a value of 0.1 will cause a solution to be optimal when the
    optimality gap is at most 10%. See also `absolute_gap_threshold` for a
    non-relative variant.
    """

    absolute_gap_threshold: Optional[float] = None
    """Absolute gap threshold below which a solution is considered optimal

    See also `relative_gap_threshold` for a relative variant.
    """

    zero_value_threshold: Optional[float] = None
    """Positive magnitude below which tensor values are assumed equal to zero

    This option is also used on solution results, causing values to be omitted
    from the solution if their dual value is also absent. It is finally used as
    threshold for rounding integral variables to the nearest integer. The
    default is 1e-6.
    """

    infinity_value_threshold: Optional[float] = None
    """Positive magnitude used to cap all input values

    It is illegal for the reified problem to include coefficients higher or
    equal to this value so the input needs to be such that they are masked out
    during reification. The default is 1e13.
    """

    free_bound_threshold: Optional[float] = None
    """Positive magnitude used to decide whether a bound is free

    This value should typically be slightly smaller to the infinity value
    threshold to allow for small offsets to infinite values. The default is
    1e12.
    """

    timeout_millis: Optional[float] = None
    """Upper bound on solving time"""


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
            "relaxation": _relaxation_to_json(relaxation)
            if relaxation
            else None,
        }
    )
