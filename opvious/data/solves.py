from __future__ import annotations

import collections
import dataclasses
import pandas as pd
from typing import Any, cast, Mapping, Optional, Union

from ..common import decode_extended_float, Json, json_dict
from .outcomes import (
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
    Outcome,
    SolveStatus,
)
from .outlines import Label, ObjectiveSense, Outline


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
                    data=(e["value"] for e in entries),
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

    def variable(self, label: Label) -> pd.DataFrame:
        """Returns variable results for a given label

        The returned dataframe has two columns: `value` and `dual_value`.
        """
        for res in self.raw_variables:
            if res["label"] == label:
                entries = res["entries"]
                bindings = self.outline.variables[label].bindings
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


def solve_options_to_json(options: Optional[SolveOptions] = None) -> Json:
    if not options:
        return None
    return json_dict(**dataclasses.asdict(options or SolveOptions()))


Target = Union[Label, Mapping[Label, float]]
"""Target objective

A single label is equivalent to optimizing just the objective with that label
and ignoring all others. If using a mapping, all objective keys must have an
associated values.
"""


def _target_to_json(target: Target, outline: Outline) -> Json:
    if isinstance(target, str):
        target = collections.defaultdict(lambda: 0, {target: 1})
    unknown = target.keys() - outline.objectives.keys()
    if unknown:
        raise Exception(f"Unknown objective(s): {unknown}")
    weights = [
        {"label": label, "value": target[label]}
        for label in outline.objectives
    ]
    return json_dict(weights=weights)


@dataclasses.dataclass(frozen=True)
class EpsilonConstraint:
    """Constraint enforcing proximity to a objective's optimal value"""

    target: Target
    """Target objective"""

    absolute_tolerance: Optional[float] = None
    """Cap on the absolute value of the final solution vs optimal"""

    relative_tolerance: Optional[float] = None
    """Cap on the relative value of the final solution vs optimal"""


@dataclasses.dataclass(frozen=True)
class SolveStrategy:
    """Multi-objective solving strategy"""

    target: Target
    """Target objective"""

    sense: Optional[ObjectiveSense] = None
    """Optimization sense"""

    epsilon_constraints: list[EpsilonConstraint] = dataclasses.field(
        default_factory=lambda: []
    )
    """All epsilon-constraints to apply"""

    @classmethod
    def equally_weighted_sum(cls, sense: Optional[ObjectiveSense] = None):
        """Returns a strategy optimizing the sum of all objectives"""
        return SolveStrategy(
            target=collections.defaultdict(lambda: 1), sense=sense
        )


def solve_strategy_to_json(
    strategy: Optional[SolveStrategy], outline: Outline
) -> Json:
    if not strategy:
        return None
    target = strategy.target
    if isinstance(target, str):
        target = collections.defaultdict(lambda: 0, {target: 1})
    sense = strategy.sense
    if not sense:
        for label, objective in outline.objectives.items():
            weight = target[label]
            if not weight:
                continue
            if sense is None:
                sense = objective.sense
            elif sense != objective.sense:
                raise Exception("Explicit objective sense required")
        if not sense:
            raise Exception("Missing objective")
    return json_dict(
        is_maximization=sense == "MAXIMIZE",
        target=_target_to_json(target, outline),
        epsilon_constraints=[
            json_dict(
                relative_tolerance=c.relative_tolerance,
                absolute_tolerance=c.absolute_tolerance,
                target=_target_to_json(c.target, outline),
            )
            for c in strategy.epsilon_constraints
        ],
    )
