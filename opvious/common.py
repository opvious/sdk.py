from importlib import metadata
import math
from typing import Any, Iterable, Union
import urllib.parse


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""


del metadata


# Formatting


Label = str


def format_percent(val):
    if val == "Infinity":
        return "inf"
    return f"{int(val * 10_000) / 100}%"


def is_url(s: str) -> bool:
    """Checks if a string is a URL."""
    try:
        res = urllib.parse.urlparse(s)
        return bool(res.scheme and res.netloc)
    except ValueError:
        return False


def to_camel_case(s: str) -> str:
    if "_" not in s:
        return s
    return "".join(
        p.capitalize() if i else p for i, p in enumerate(s.split("_")) if p
    )


def untuple(t: Iterable[Any]) -> Any:
    if not isinstance(t, tuple):
        t = tuple(t)
    return t[0] if len(t) == 1 else t


# JSON utilities


Json = Any


ExtendedFloat = Union[float, str]


def encode_extended_float(val: ExtendedFloat) -> Json:
    if val == math.inf:
        return "Infinity"
    elif val == -math.inf:
        return "-Infinity"
    return val


def decode_extended_float(val: ExtendedFloat) -> Json:
    if val == "Infinity":
        return math.inf
    elif val == "-Infinity":
        return -math.inf
    return val


def json_dict(**kwargs) -> Json:
    """Strips keys with None values and encodes infinity values"""
    data = {}
    for key, val in kwargs.items():
        if val is None:
            continue
        json_key = to_camel_case(key)
        if isinstance(val, float):
            data[json_key] = encode_extended_float(val)
        else:
            data[json_key] = val
    return data
