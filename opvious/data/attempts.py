from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Optional

from .outlines import Outline
from .solves import SolveInputs
from .tensors import Value


@dataclasses.dataclass(frozen=True)
class AttemptRequest:
    """Attempt creation request"""

    formulation_name: str
    """The underlying formulation's name"""

    specification_tag_name: str
    """The target specification tag"""

    inputs: SolveInputs = dataclasses.field(repr=False)
    """Input data"""


@dataclasses.dataclass(frozen=True)
class Attempt:
    """Queueable optimization attempt"""

    uuid: str
    """The attempt's unique identifier"""

    started_at: datetime
    """The time the attempt was created"""

    outline: Outline = dataclasses.field(repr=False)
    """The specification outline corresponding to this attempt"""

    url: str = dataclasses.field(repr=False)
    """The optimimization hub URL for this attempt"""


def attempt_from_graphql(data: Any, outline: Outline, url: str) -> Attempt:
    return Attempt(
        uuid=data["uuid"],
        started_at=datetime.fromisoformat(data["startedAt"]),
        outline=outline,
        url=url,
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
