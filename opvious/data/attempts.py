from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Optional

from .outlines import Outline
from .tensors import Value


@dataclasses.dataclass(frozen=True)
class Attempt:
    """Queueable optimization attempt

    New attempts are started via :meth:`.Client.start_attempt`, existing
    attempts can be retrieved from their UUID via :meth:`.Client.load_attempt`.
    """

    uuid: str
    """The attempt's unique identifier"""

    started_at: datetime
    """The time the attempt was created"""

    outline: Outline = dataclasses.field(repr=False)
    """The specification outline corresponding to this attempt"""


def attempt_from_graphql(data: Any, outline: Outline) -> Attempt:
    return Attempt(
        uuid=data["uuid"],
        started_at=datetime.fromisoformat(data["startedAt"]),
        outline=outline,
    )


@dataclasses.dataclass(frozen=True)
class AttemptNotification:
    """Attempt progress update notification"""

    dequeued: bool
    """Whether the attempt has already been dequeued"""

    relative_gap: Optional[Value]
    """The latest relative gap"""

    lp_iteration_count: Optional[int]
    """The latest LP iteration count"""

    cut_count: Optional[int]
    """The latest cut count"""


def notification_from_graphql(
    dequeued: bool, data: Any = None
) -> AttemptNotification:
    return AttemptNotification(
        dequeued=dequeued,
        relative_gap=data["relativeGap"] if data else None,
        lp_iteration_count=data["lpIterationCount"] if data else None,
        cut_count=data["cutCount"] if data else None,
    )
