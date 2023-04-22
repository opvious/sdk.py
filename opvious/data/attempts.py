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
from datetime import datetime
from typing import Any, Optional

from .outlines import Outline
from .solves import SolveInputs
from .tensors import Value


@dataclasses.dataclass(frozen=True)
class AttemptRequest:
    """Attempt inputs"""

    formulation_name: str
    specification_tag_name: str
    inputs: SolveInputs = dataclasses.field(repr=False)


@dataclasses.dataclass(frozen=True)
class Attempt:
    uuid: str
    started_at: datetime
    url: str = dataclasses.field(repr=False)
    outline: Outline = dataclasses.field(repr=False)

    @classmethod
    def from_graphql(cls, data: Any, outline: Outline, url: str):
        return Attempt(
            uuid=data["uuid"],
            started_at=datetime.fromisoformat(data["startedAt"]),
            outline=outline,
            url=url,
        )


@dataclasses.dataclass(frozen=True)
class Notification:
    dequeued: bool
    relative_gap: Optional[Value]
    lp_iteration_count: Optional[int]
    cut_count: Optional[int]

    @classmethod
    def from_graphql(cls, dequeued: bool, data: Any = None):
        return Notification(
            dequeued=dequeued,
            relative_gap=data["relativeGap"] if data else None,
            lp_iteration_count=data["lpIterationCount"] if data else None,
            cut_count=data["cutCount"] if data else None,
        )
