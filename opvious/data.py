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

import dataclasses
from typing import Any, Optional, Union


KeyItem = Union[float, str]


Key = tuple[KeyItem, ...]


Label = str


@dataclasses.dataclass
class Inputs:
    formulation_name: str
    tag_name: str
    dimensions: list[Any]
    parameters: list[Any]


@dataclasses.dataclass
class FailedOutcome:
    status: str
    message: str
    code: Optional[str]
    tags: Any


@dataclasses.dataclass
class FeasibleOutcome:
    is_optimal: bool
    objective_value: float
    relative_gap: Optional[float]


@dataclasses.dataclass
class InfeasibleOutcome:
    pass


@dataclasses.dataclass
class UnboundedOutcome:
    pass


Outcome = Union[
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
]
