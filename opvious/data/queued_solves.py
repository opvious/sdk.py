from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Mapping, Optional

from .outlines import ProblemOutline
from .tensors import Value


AttemptAttributes = Mapping[str, str]


@dataclasses.dataclass(frozen=True)
class QueuedSolve:
    """Queued optimization attempt

    Solves are queued via :meth:`.Client.queue_solve`, existing queued solves
    can be retrieved from their UUID via :meth:`.Client.fetch_solve`.
    """

    uuid: str
    """The solve's unique identifier"""

    outline: ProblemOutline = dataclasses.field(repr=False)
    """The specification outline corresponding to this solve"""

    started_at: datetime
    """The time the solve was created"""


def queued_solve_from_graphql(
    data: Any, outline: ProblemOutline
) -> QueuedSolve:
    return QueuedSolve(
        uuid=data["uuid"],
        outline=outline,
        started_at=datetime.fromisoformat(data["attempt"]["startedAt"]),
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
