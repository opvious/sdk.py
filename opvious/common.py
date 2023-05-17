import enum
from importlib import metadata
import math
import os
from typing import Any, Optional, Union, cast
import urllib.parse


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""


del metadata


# Configuration


_DEFAULT_DOMAIN = "beta.opvious.io"


class Setting(enum.Enum):
    """Client configuration environment variables"""

    TOKEN = ("OPVIOUS_TOKEN", "")
    DOMAIN = ("OPVIOUS_DOMAIN", _DEFAULT_DOMAIN)

    def read(self, env: Optional[dict[str, str]] = None) -> str:
        """Read the setting's current value or default if missing

        Args:
            env: Environment, defaults to `os.environ`.
        """
        if env is None:
            env = cast(Any, os.environ)
        name, default_value = self.value
        return env.get(name) or default_value


def default_api_url(domain: Optional[str] = None) -> str:
    return f"https://api.{domain or _DEFAULT_DOMAIN}"


def default_hub_url(domain: Optional[str] = None) -> str:
    return f"https://hub.{domain or _DEFAULT_DOMAIN}"


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
        p.capitalize() if i else p for i, p in enumerate(s.split("_"))
    )


def untuple(t: tuple[Any, ...]) -> Any:
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
