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

KeyItem = Union[float, str]

Key = tuple[KeyItem, ...]

@dataclasses.dataclass
class ParameterEntry:
  key: Key
  value: float

@dataclasses.dataclass
class Parameter:
  label: str
  entries: list[ParameterEntry]
  default_value: Optional[float] = None

  def to_input(self):
    obj = {
      'label': self.label,
      'entries': [dataclasses.asdict(e) for e in self.entries],
    }
    if self.default_value is not None:
      obj['defaultValue'] = self.default_value
    return obj

  @classmethod
  def scalar(cls, label, value):
    return Parameter(label=label, entries=[ParameterEntry(key=[], value=value)])

  @classmethod
  def indexed(cls, label, mapping, default_value=None):
    entries = [ParameterEntry(key, value) for key, value in mapping.items()]
    return Parameter(label=label, entries=entries, default_value=default_value)

  @classmethod
  def indicator(cls, label, keys):
    entries = [ParameterEntry(key, 1) for key in keys]
    return Parameter(label=label, entries=entries)

@dataclasses.dataclass
class Collection:
  label: str
  items: list[KeyItem]

  @classmethod
  def iterable(cls, label, iterable):
    return Collection(label=label, items=list(iterable))

  def to_input(self):
    return dataclasses.asdict(self)

@dataclasses.dataclass
class FailedOutcome:
  error_messages: list[str]

@dataclasses.dataclass
class IndexedVariable:
  label: str
  value: Mapping[Key, float]

@dataclasses.dataclass
class ScalarVariable:
  label: str
  value: float

Variable = Union[ScalarVariable, IndexedVariable]

@dataclasses.dataclass
class FeasibleOutcome:
  is_optimal: bool
  objective_value: float
  relative_gap: float
  variables: list[Variable]

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
