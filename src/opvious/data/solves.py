from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any, Self, cast

import numpy as np
import pandas as pd

from ..common import Json, decode_extended_float, json_dict
from .outcomes import (
    AbortedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    SolveOutcome,
    SolveStatus,
    UnboundedOutcome,
)
from .outlines import Label, ObjectiveSense, ProblemOutline, SourceBinding
from .tensors import KeyItem


@dataclasses.dataclass(frozen=True)
class ProblemSummary:
    """Reified problem summary statistics"""

    column_count: int
    """Total number of variable columns"""

    row_count: int
    """Total number of constraint rows"""

    dimensions: pd.DataFrame = dataclasses.field(repr=False)
    """Dimension summary statistics"""

    parameters: pd.DataFrame = dataclasses.field(repr=False)
    """Parameter summary statistics"""

    variables: pd.DataFrame = dataclasses.field(repr=False)
    """Variable summary statistics"""

    constraints: pd.DataFrame = dataclasses.field(repr=False)
    """Constraint summary statistics"""

    objectives: pd.DataFrame = dataclasses.field(repr=False)
    """Objective summary statistics"""


def problem_summary_from_json(data: Json) -> ProblemSummary:
    return ProblemSummary(
        column_count=sum(v["columnCount"] for v in data["variables"]),
        row_count=sum(c["rowCount"] for c in data["constraints"]),
        dimensions=_labeled_dataframe(
            {"label": c["label"], "item_count": c["itemCount"]}
            for c in data["dimensions"]
        ),
        parameters=_labeled_dataframe(
            (
                {
                    "label": c["label"],
                    **_value_profile("entry", c["entryProfile"]),
                    f"entry_{_MULTIPLICITY_SUFFIX}": c["domainMultiplicity"],
                }
                for c in data["parameters"]
            ),
            multiplicities=["entry"],
        ),
        variables=_labeled_dataframe(
            (
                {
                    "label": c["label"],
                    "column_count": c["columnCount"],
                    f"column_{_MULTIPLICITY_SUFFIX}": c["domainMultiplicity"],
                }
                for c in data["variables"]
            ),
            multiplicities=["column"],
        ),
        constraints=_labeled_dataframe(
            (
                {
                    "label": c["label"],
                    "row_count": c["rowCount"],
                    f"row_{_MULTIPLICITY_SUFFIX}": c["domainMultiplicity"],
                    f"row_{_SPARSITY}": math.nan,
                    "column_count": c["columnCount"],
                    f"column_{_MULTIPLICITY_SUFFIX}": c[
                        "coefficientMultiplicity"
                    ],
                    f"column_{_SPARSITY}": math.nan,
                    **_value_profile("weight", c["weightProfile"]),
                    f"weight_{_MULTIPLICITY_SUFFIX}": int(
                        c["domainMultiplicity"]
                    )
                    * int(c["coefficientMultiplicity"]),
                    f"weight_{_SPARSITY}": math.nan,
                    "reify_ms": _timedelta(c["reifiedInMillis"]),
                }
                for c in data["constraints"]
            ),
            multiplicities=["row", "column", "weight"],
        ),
        objectives=_labeled_dataframe(
            (
                {
                    "label": c["label"],
                    **_value_profile("weight", c["weightProfile"]),
                    f"weight_{_MULTIPLICITY_SUFFIX}": c[
                        "coefficientMultiplicity"
                    ],
                    f"weight_{_SPARSITY}": math.nan,
                    "reify_ms": _timedelta(c["reifiedInMillis"]),
                }
                for c in data["objectives"]
            ),
            multiplicities=["weight"],
        ),
    )


_MULTIPLICITY_SUFFIX = "mult"


_SPARSITY = "sprs"


def _labeled_dataframe(
    gen: Any, multiplicities: Sequence[str] | None = None
) -> pd.DataFrame:
    df = pd.DataFrame(gen)
    if not len(df):
        return df
    for k in multiplicities or []:
        m = f"{k}_{_MULTIPLICITY_SUFFIX}"
        se = pd.to_numeric(df[m])
        del df[m]
        with np.errstate(divide="ignore"):
            df[f"{k}_{_SPARSITY}"] = -np.log(df[f"{k}_count"] / se) + 0
    return df.set_index("label")


def _timedelta(ms: int) -> int:
    return ms


_value_profile_keys = ["count", "min", "max", "mean", "stddev"]


def _value_profile(prefix: str, profile: Json) -> Mapping[str, float]:
    return {
        f"{prefix}_{k}": profile.get(k, math.nan) for k in _value_profile_keys
    }


def _entry_index(
    entries: Sequence[Json], bindings: Sequence[SourceBinding]
) -> pd.Index | pd.MultiIndex | None:
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

    problem_outline: ProblemOutline
    """Target model metadata"""

    raw_parameters: Sequence[Json] = dataclasses.field(repr=False)
    """All parameters in raw format"""

    raw_dimensions: Sequence[Json] | None = dataclasses.field(repr=False)
    """All dimensions in raw format"""

    def parameter(
        self,
        label: Label,
        coerce: bool = True,
        index: pd.Index | Sequence[KeyItem] | None = None,
    ) -> pd.DataFrame:
        """Returns the parameter for a given label as a pandas dataframe

        The returned dataframe has a `value` column with the parameter's values
        (0 values may be omitted).

        Args:
            label: Parameter label to retrieve
            coerce: Round integral parameters
            index: Returned dataframe index
        """
        for param in self.raw_parameters:
            if param["label"] == label:
                outline = self.problem_outline.parameters[label]
                return _tensor_json_dataframe(
                    param,
                    outline.bindings,
                    index=index,
                    round_values=coerce and outline.is_integral,
                )
        raise Exception(f"Unknown parameter: {label}")

    def dimension(self, label: Label) -> pd.Index:
        """Returns the dimension for a given label as a pandas Index"""
        if self.raw_dimensions is not None:
            for dim in self.raw_dimensions:
                if dim["label"] == label:
                    return pd.Index(dim["items"], name=label)
        else:
            items = set()
            has_binding = False
            for param in self.raw_parameters:
                outline = self.problem_outline.parameters[param["label"]]
                for i, binding in enumerate(outline.bindings):
                    if binding.dimension_label != label:
                        continue
                    has_binding = True
                    for entry in param["entries"]:
                        items.add(entry["key"][i])
            if has_binding:
                return pd.Index(items, name=label).sort_values()
        raise Exception(f"Unknown dimension: {label}")


@dataclasses.dataclass(frozen=True)
class SolveOutputs:
    """Successful solve output data"""

    problem_outline: ProblemOutline
    """Solved model metadata"""

    raw_variables: Sequence[Json] = dataclasses.field(repr=False)
    """All variables in raw format"""

    raw_constraints: Sequence[Json] = dataclasses.field(repr=False)
    """All constraints in raw format"""

    def variable(
        self,
        label: Label,
        coerce: bool = True,
        index: pd.Index | Sequence[KeyItem] | None = None,
    ) -> pd.DataFrame:
        """Returns variable results for a given label

        The returned dataframe always has a `value` column with the variable's
        values (0 values may be omitted). If applicable, it will also have a
        `reduced_cost` column.

        Args:
            label: Variable label to retrieve
            coerce: Round integral variables
            index: Returned dataframe index
        """
        for res in self.raw_variables:
            if res["label"] == label:
                outline = self.problem_outline.variables[label]
                return _tensor_json_dataframe(
                    res,
                    outline.bindings,
                    dual_value_name="reduced_cost",
                    index=index,
                    round_values=coerce and outline.is_integral,
                )
        raise Exception(f"Unknown variable {label}")

    def constraint(self, label: Label) -> pd.DataFrame:
        """Returns constraint results for a given label.

        The returned dataframe always has a `slack` column with the
        constraint's slack (0 values may be omitted). If applicable, it will
        also have a `shadow_price` column.
        """
        for res in self.raw_constraints:
            if res["label"] == label:
                return _tensor_json_dataframe(
                    res,
                    self.problem_outline.constraints[label].bindings,
                    value_name="slack",
                    dual_value_name="shadow_price",
                )
        raise Exception(f"Unknown constraint {label}")


def _tensor_json_dataframe(
    tensor_json: Json,
    bindings: Sequence[SourceBinding],
    *,
    value_name: str = "value",
    dual_value_name: str | None = None,
    index: pd.Index | Sequence[KeyItem] | None = None,
    round_values: bool = False,
) -> pd.DataFrame:
    entries = tensor_json["entries"]
    default_values = {
        value_name: decode_extended_float(tensor_json.get("defaultValue", 0)),
    }
    if dual_value_name:
        data = (
            (decode_extended_float(e["value"]), e.get("dualValue"))
            for e in entries
        )
        columns = [value_name, dual_value_name]
        default_values[dual_value_name] = 0
    else:
        data = (decode_extended_float(e["value"]) for e in entries)
        columns = [value_name]
    df = pd.DataFrame(
        data=data,
        columns=columns,
        index=_entry_index(entries, bindings),
    )
    if dual_value_name and df[dual_value_name].isnull().all():
        df = df.drop(dual_value_name, axis=1)
    df = df.sort_index() if index is None else df.reindex(cast(Any, index))
    df = df.fillna(default_values)
    if round_values:
        df[value_name] = df[value_name].round(0).astype(np.int64)
    return df


def _outputs_from_json(data: Json, outline: ProblemOutline) -> SolveOutputs:
    return SolveOutputs(
        problem_outline=outline,
        raw_variables=data["variables"],
        raw_constraints=data["constraints"],
    )


@dataclasses.dataclass(frozen=True)
class Solution:
    """Solver response"""

    status: SolveStatus
    """Status string"""

    outcome: SolveOutcome
    """Solution metadata"""

    problem_summary: ProblemSummary
    """Problem summary statistics"""

    inputs: SolveInputs = dataclasses.field(repr=False)
    """Problem inputs"""

    outputs: SolveOutputs | None = dataclasses.field(default=None, repr=False)
    """Solution data, present iff the solution is feasible"""

    @property
    def feasible(self) -> bool:
        """Returns true iff the solution's outcome is feasible"""
        return isinstance(self.outcome, FeasibleOutcome)


def solution_from_json(
    outline: ProblemOutline,
    inputs: SolveInputs,
    response_json: Any,
    problem_summary: ProblemSummary | None = None,
) -> Solution:
    outcome_json = response_json["outcome"]
    match status := outcome_json["status"]:
        case "INFEASIBLE":
            outcome = cast(SolveOutcome, InfeasibleOutcome())
        case "UNBOUNDED":
            outcome = UnboundedOutcome()
        case "ABORTED":
            outcome = AbortedOutcome()
        case _:
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
    return Solution(
        status=status,
        outcome=outcome,
        problem_summary=problem_summary
        or problem_summary_from_json(response_json["summaries"]["problem"]),
        inputs=inputs,
        outputs=outputs,
    )


@dataclasses.dataclass(frozen=True)
class SolveOptions:
    """Solving options"""

    relative_gap_threshold: float | None = None
    """Relative gap threshold below which a solution is considered optimal

    For example a value of 0.1 will cause a solution to be optimal when the
    optimality gap is at most 10%. See also `absolute_gap_threshold` for a
    non-relative variant.
    """

    absolute_gap_threshold: float | None = None
    """Absolute gap threshold below which a solution is considered optimal

    See also `relative_gap_threshold` for a relative variant.
    """

    zero_value_threshold: float | None = None
    """Positive magnitude below which tensor values are assumed equal to zero

    This option is also used on solution results, causing values to be omitted
    from the solution if their dual value is also absent. It is finally used as
    threshold for rounding integral variables to the nearest integer. The
    default is 1e-6.
    """

    infinity_value_threshold: float | None = None
    """Positive magnitude used to cap all input values

    It is illegal for the reified problem to include coefficients higher or
    equal to this value so the input needs to be such that they are masked out
    during reification. The default is 1e13.
    """

    free_bound_threshold: float | None = None
    """Positive magnitude used to decide whether a bound is free

    This value should typically be slightly smaller to the infinity value
    threshold to allow for small offsets to infinite values. The default is
    1e12.
    """

    timeout_millis: float | None = None
    """Upper bound on solving time"""


def solve_options_to_json(options: SolveOptions | None = None) -> Json:
    if not options:
        return None
    return json_dict(**dataclasses.asdict(options or SolveOptions()))


type Target = Label | Mapping[Label, float]
"""Target objective

A single label is equivalent to optimizing just the objective with that label
and ignoring all others. If using a mapping, all objective keys must have an
associated values.
"""


def _target_to_json(target: Target, outline: ProblemOutline) -> Json:
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

    absolute_tolerance: float | None = None
    """Cap on the absolute value of the final solution vs optimal"""

    relative_tolerance: float | None = None
    """Cap on the relative value of the final solution vs optimal"""


@dataclasses.dataclass(frozen=True)
class SolveStrategy:
    """Multi-objective solving strategy"""

    target: Target
    """Target objective"""

    sense: ObjectiveSense | None = None
    """Optimization sense"""

    epsilon_constraints: Sequence[EpsilonConstraint] = dataclasses.field(
        default_factory=lambda: []
    )
    """All epsilon-constraints to apply"""

    @classmethod
    def equally_weighted_sum(cls, sense: ObjectiveSense | None = None) -> Self:
        """Returns a strategy optimizing the sum of all objectives"""
        return cls(target=collections.defaultdict(lambda: 1), sense=sense)


def solve_strategy_to_json(
    strategy: SolveStrategy | None, outline: ProblemOutline
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
