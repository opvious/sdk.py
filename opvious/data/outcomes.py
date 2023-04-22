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
from typing import Any, Optional, Union

from .tensors import Value


SolveStatus = str


@dataclasses.dataclass(frozen=True)
class CancelledOutcome:
    """The solve was cancelled before a solution was found"""


@dataclasses.dataclass(frozen=True)
class FailedOutcome:
    """The solve failed"""

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


@dataclasses.dataclass(frozen=True)
class FeasibleOutcome:
    """A solution was found"""

    is_optimal: bool
    objective_value: Optional[Value]
    relative_gap: Optional[Value]

    @classmethod
    def from_graphql(cls, data: Any) -> FeasibleOutcome:
        return FeasibleOutcome(
            is_optimal=data["isOptimal"],
            objective_value=data.get("objectiveValue"),
            relative_gap=data.get("relativeGap"),
        )


@dataclasses.dataclass(frozen=True)
class InfeasibleOutcome:
    """No feasible solution exists"""


@dataclasses.dataclass(frozen=True)
class UnboundedOutcome:
    """No bounded optimal solution exists"""


Outcome = Union[
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
]


def outcome_status(outcome: Outcome) -> SolveStatus:
    """Returns the status corresponding to a given outcome"""
    if isinstance(outcome, CancelledOutcome):
        return "CANCELLED"
    if isinstance(outcome, FailedOutcome):
        return "ERRORED"
    if isinstance(outcome, FeasibleOutcome):
        return "OPTIMAL" if outcome.is_optimal else "FEASIBLE"
    if isinstance(outcome, InfeasibleOutcome):
        return "INFEASIBLE"
    if isinstance(outcome, UnboundedOutcome):
        return "INFEASIBLE"
    raise TypeError(f"Unexpected outcome: {outcome}")


class UnexpectedOutcomeError(Exception):
    """The solve ended with an unexpected outcome"""

    def __init__(self, outcome: Outcome):
        self.outcome = outcome
