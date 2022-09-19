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
import pandas as pd
from typing import Mapping, Optional, Union

@dataclasses.dataclass
class Definition:
  kind: str
  source: str
  label: Optional[str]
  description: Optional[str]

@dataclasses.dataclass
class Formulation:
  name: str
  display_name: str
  description: str
  url: Optional[str]
  created_at: str

KeyItem = Union[float, str]

Key = tuple[KeyItem, ...]

@dataclasses.dataclass
class ParameterEntry:
  key: Key
  value: float

  def __init__(self, key, value):
    self.key = tuple(key) if isinstance(key, (list, tuple)) else (key,)
    self.value = value

@dataclasses.dataclass
class Parameter:
  label: str
  entries: list[ParameterEntry]
  default_value: float = 0

  def to_input(self):
    return {
      'label': self.label,
      'entries': [dataclasses.asdict(e) for e in self.entries],
      'defaultValue': self.default_value,
    }

  @classmethod
  def scalar(cls, label, value):
    return Parameter(label=label, entries=[ParameterEntry(key=[], value=value)])

  @classmethod
  def indexed(cls, label, mapping, default_value=0):
    entries = [ParameterEntry(key, value) for key, value in mapping.items()]
    return Parameter(label=label, entries=entries, default_value=default_value)

  @classmethod
  def indicator(cls, label, keys):
    entries = [ParameterEntry(key, 1) for key in keys]
    return Parameter(label=label, entries=entries)

@dataclasses.dataclass
class Dimension:
  label: str
  items: list[KeyItem]

  @classmethod
  def iterable(cls, label, iterable):
    return Dimension(label=label, items=list(iterable))

  def to_input(self):
    return dataclasses.asdict(self)

@dataclasses.dataclass
class FailedOutcome:
  status: str
  message: str
  code: Optional[str]
  tags: any

@dataclasses.dataclass
class IndexedResult:
  label: str
  value: Mapping[Key, float]

@dataclasses.dataclass
class ScalarResult:
  label: str
  value: float

Result = Union[ScalarResult, IndexedResult]

@dataclasses.dataclass
class FeasibleOutcome:
  is_optimal: bool
  objective_value: float
  absolute_gap: Optional[float]
  variable_results: list[Result]

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

@dataclasses.dataclass
class AttemptTemplate:
  dimensions: list[Dimension]
  parameters: list[Parameter]
