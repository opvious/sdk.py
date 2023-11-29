from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional, Union

from .tensors import Value


SolveStatus = Literal[
    "ABORTED",
    "FEASIBLE",
    "INFEASIBLE",
    "OPTIMAL",
    "UNBOUNDED",
    "UNKNOWN",
]


@dataclasses.dataclass(frozen=True)
class AbortedOutcome:
    """The solve was cancelled before a solution was found"""


@dataclasses.dataclass(frozen=True)
class FailedOutcome:
    """The solve failed"""

    status: str
    """The underlying error's status"""

    message: str
    """The underlying error's message"""

    code: Optional[str] = None
    """The underlying error's error code"""

    tags: Any = None
    """Structured data associated with the failure"""


def failed_outcome_from_graphql(data: Any) -> FailedOutcome:
    error = data["error"]
    return FailedOutcome(
        status=data["status"],
        message=error["message"],
        code=error.get("code"),
        tags=error.get("tags"),
    )


@dataclasses.dataclass(frozen=True)
class FeasibleOutcome:
    """A solution was found"""

    optimal: bool
    """Whether this solution was optimal (within gap thresholds)"""

    objective_value: Optional[Value]
    """The solution's objective value"""

    relative_gap: Optional[Value]
    """The solution's relative gap (0.1 is 10%)"""


def feasible_outcome_from_graphql(data: Any) -> FeasibleOutcome:
    return FeasibleOutcome(
        optimal=data["status"] == "OPTIMAL",
        objective_value=data.get("objectiveValue"),
        relative_gap=data.get("relativeGap"),
    )


@dataclasses.dataclass(frozen=True)
class InfeasibleOutcome:
    """No feasible solution exists"""


@dataclasses.dataclass(frozen=True)
class UnboundedOutcome:
    """No bounded optimal solution exists"""


SolveOutcome = Union[
    AbortedOutcome,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
]


def solve_outcome_status(outcome: SolveOutcome) -> SolveStatus:
    """Returns the status corresponding to a given outcome"""
    if isinstance(outcome, AbortedOutcome):
        return "ABORTED"
    if isinstance(outcome, FeasibleOutcome):
        return "OPTIMAL" if outcome.optimal else "FEASIBLE"
    if isinstance(outcome, InfeasibleOutcome):
        return "INFEASIBLE"
    if isinstance(outcome, UnboundedOutcome):
        return "INFEASIBLE"
    return "UNKNOWN"


class UnexpectedSolveOutcomeError(Exception):
    """The solve ended with an unexpected outcome"""

    def __init__(self, outcome: SolveOutcome):
        self.outcome = outcome
