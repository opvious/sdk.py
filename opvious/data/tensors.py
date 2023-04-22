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
import math
import pandas as pd
from typing import Any, Iterable, Mapping, Tuple, Union


KeyItem = Union[float, int, str]


Key = Union[Tuple[KeyItem, ...], KeyItem]


Value = Union[float, int]


def is_value(arg: Any) -> bool:
    return isinstance(arg, (float, int))


DimensionArgument = Iterable[KeyItem]


SparseTensorArgument = Union[
    pd.Series,
    Mapping[Key, Value],
    pd.DataFrame,  # For indicator parameters
    Iterable[Key],  # For indicator parameters
]


TensorArgument = Union[
    Value, SparseTensorArgument, Tuple[SparseTensorArgument, Value]
]


ExtendedFloat = Union[float, str]


def encode_extended_float(val: float):
    if val == math.inf:
        return "Infinity"
    elif val == -math.inf:
        return "-Infinity"
    return val


def decode_extended_float(val: ExtendedFloat):
    if val == "Infinity":
        return math.inf
    elif val == "-Infinity":
        return -math.inf
    return val


@dataclasses.dataclass
class Tensor:
    entries: list[Any]
    default_value: ExtendedFloat = 0

    @classmethod
    def from_argument(
        cls, arg: TensorArgument, rank: int, is_indicator: bool = False
    ):
        if isinstance(arg, tuple):
            data, default_value = arg
        elif rank > 0 and is_value(arg) and not is_indicator:
            data = {}
            default_value = arg
        else:
            data = arg
            default_value = 0
        if (
            is_indicator
            and isinstance(data, pd.Series)
            and not pd.api.types.is_numeric_dtype(data)
        ):
            data = data.reset_index()
        keyifier = _Keyifier(rank)
        if is_indicator and isinstance(data, pd.DataFrame):
            entries = [
                {"key": keyifier.keyify(key), "value": 1}
                for key in data.itertuples(index=False, name=None)
            ]
        elif is_indicator and not hasattr(data, "items"):
            entries = [
                {"key": keyifier.keyify(key), "value": 1} for key in data
            ]
        else:
            if is_value(data):
                entries = [{"key": (), "value": encode_extended_float(data)}]
            else:
                entries = [
                    {
                        "key": keyifier.keyify(key),
                        "value": encode_extended_float(value),
                    }
                    for key, value in data.items()
                ]
        return Tensor(entries, encode_extended_float(default_value))


class _Keyifier:
    def __init__(self, rank: int):
        self.rank = rank

    def keyify(self, key):
        tup = tuple(key) if isinstance(key, (list, tuple)) else (key,)
        if len(tup) != self.rank:
            raise Exception(f"Invalid key rank: {len(tup)} != {self.rank}")
        return tup
