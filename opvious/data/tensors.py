from __future__ import annotations

import dataclasses
import math
import pandas as pd
from typing import Any, Iterable, Mapping, Tuple, Union

from ..common import ExtendedFloat, encode_extended_float


KeyItem = Union[float, int, str]


Key = Union[Tuple[KeyItem, ...], KeyItem]


Value = Union[float, int]


def is_value(arg: Any) -> bool:
    return isinstance(arg, (float, int))


DimensionArgument = Iterable[KeyItem]


SparseTensorArgument = Union[
    Mapping[Key, Value],
    pd.Series,
    pd.DataFrame,  # For indicator parameters
    Iterable[Key],  # For indicator parameters
]


#: Input tensor creation argument
TensorArgument = Union[
    Value, SparseTensorArgument, Tuple[SparseTensorArgument, Value]
]


@dataclasses.dataclass
class Tensor:
    """An n-dimensional matrix"""

    entries: list[Any]
    """Raw list of matrix entries"""

    default_value: ExtendedFloat = 0
    """Value to use for missing key"""

    @classmethod
    def from_argument(
        cls,
        arg: TensorArgument,
        rank: int,
        is_indicator: bool = False,
        is_pin: bool = False,
    ):
        """Creates a tensor from a variety of argument values

        In most cases you will not need to call this method directly: it is
        called automatically during parameter validation.

        Args:
            arg: Raw argument to be wrapped into a tensor. This argument must
                match the tensor's shape, see below for details.
            rank: The expected rank of the tensor
            is_indicator: Whether the tensor holds indicator values

        The accepted arguments depend on the tensor's domain rank (the length
        of its keys) and image (does it hold arbitrary numbers or only 0s/1s).
        If the tensor's rank is 0, i.e. is a scalar, the only accepted argument
        type is a number matching the image. Otherwise the following inputs are
        accepted:

        + Mapping (e.g. Python dictionary) with tuple key if rank is greater
          than 1
        + `pandas` series with key index

        Additionally, if the tensor holds indicator values, two more
        conveniences are allowed, each representing the set of keys which have
        value 1:

        + Iterable of keys
        + `pandas` dataframe

        Finally, all non-scalar tensor arguments can be wrapped into a tuple
        `(arg, default)` to provide a `default` value to use when no matching
        key exists.
        """
        if isinstance(arg, tuple):
            data, default_value = arg
        elif rank > 0 and is_value(arg) and not is_indicator:
            data = {}
            default_value = arg
        else:
            data = arg
            default_value = -math.inf if is_pin else 0
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
