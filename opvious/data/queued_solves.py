from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from typing import Any, Optional, cast

from ..common import (
    Annotation,
    Json,
    Uuid,
    decode_datetime,
    decode_annotations,
    if_present,
)
from .outcomes import (
    SolveOutcome,
    failed_outcome_from_graphql,
    solve_outcome_from_graphql,
)
from .solves import ProblemSummary, problem_summary_from_json
from .tensors import Value


@dataclasses.dataclass(frozen=True)
class QueuedSolve:
    """Queued optimization attempt

    Solves are queued via :meth:`.Client.queue_solve`, existing queued solves
    can be retrieved from their UUID via :meth:`.Client.fetch_solve`.
    """

    uuid: Uuid
    """The solve's unique identifier"""

    annotations: list[Annotation]
    """Annotation metadata"""

    outcome: Optional[SolveOutcome]
    """Final solve outcome, if available"""

    enqueued_at: datetime
    """The time the solve was created"""

    dequeued_at: Optional[datetime]
    """The time the solve started running"""

    completed_at: Optional[datetime]
    """The time the solve completed"""

    problem_summary: Optional[ProblemSummary] = dataclasses.field(repr=False)
    """Summary information about the solved problem"""

    options: Json = dataclasses.field(repr=False)
    transformations: Json = dataclasses.field(repr=False)
    strategy: Json = dataclasses.field(repr=False)

    @property
    def duration(self) -> Optional[timedelta]:
        """The solve's runtime, if it is complete"""
        return (
            self.completed_at - self.enqueued_at if self.completed_at else None
        )


def queued_solve_from_graphql(
    data: Json, attempt_data: Optional[Json] = None
) -> QueuedSolve:
    attempt_data = attempt_data or data["attempt"]

    if data["failure"]:
        outcome = cast(
            SolveOutcome, failed_outcome_from_graphql(data["failure"])
        )
    elif data["outcome"]:
        outcome = solve_outcome_from_graphql(data["outcome"])
    else:
        outcome = None

    return QueuedSolve(
        uuid=data["uuid"],
        enqueued_at=decode_datetime(attempt_data["startedAt"]),
        dequeued_at=if_present(data["dequeuedAt"], decode_datetime),
        completed_at=if_present(attempt_data["endedAt"], decode_datetime),
        annotations=decode_annotations(attempt_data["annotations"]),
        options=data["options"],
        transformations=data["transformations"],
        strategy=data["strategy"],
        outcome=outcome,
        problem_summary=if_present(
            data["problemSummary"], problem_summary_from_json
        ),
    )


@dataclasses.dataclass(frozen=True)
class SolveNotification:
    """Solve progress update notification"""

    dequeued: bool
    """Whether the solve has already been dequeued"""

    relative_gap: Optional[Value]
    """The latest relative gap"""

    lp_iteration_count: Optional[int]
    """The latest LP iteration count"""

    cut_count: Optional[int]
    """The latest cut count"""


def solve_notification_from_graphql(
    dequeued: bool, data: Any = None
) -> SolveNotification:
    return SolveNotification(
        dequeued=dequeued,
        relative_gap=data["relativeGap"] if data else None,
        lp_iteration_count=data["lpIterationCount"] if data else None,
        cut_count=data["cutCount"] if data else None,
    )
