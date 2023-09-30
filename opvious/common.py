import functools
from importlib import metadata
import math
from typing import Any, Callable, Iterable, Union
import urllib.parse
import weakref


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


# Decorator utilities


def method_decorator(wrapper: Callable[..., Any]) -> Any:
    def wrap(fn):
        return BindableMethod(fn, wrapper)

    return wrap


def with_instance(consumer):
    def wrap(fn):
        return BindableMethod(fn, consumer, lazy=True)

    return wrap


class BindableMethod:
    def __init__(
        self, body: Callable[..., Any], wrapper: Callable[..., Any], lazy=False
    ) -> None:
        self._body = body
        self._wrapper = wrapper
        self._lazy = lazy
        self._bindings: Any = weakref.WeakKeyDictionary()

    def _apply(self, owner: Any, bind=True) -> Any:
        wrapper = self._wrapper(owner) if self._lazy else self._wrapper
        body = functools.partial(self._body, owner) if bind else self._body
        return wrapper(body)

    def bound_to(self, owner: Any) -> Any:
        binding = self._bindings.get(owner)
        if not binding:
            binding = self._apply(owner)
            while isinstance(binding, BindableMethod):
                binding = binding._apply(owner, False)
            self._bindings[owner] = binding
        return binding

    def __get__(self, owner: Any, _objtype=None) -> Any:
        return self.bound_to(owner)

    def __call__(self, owner, *args, **kwargs) -> Any:
        # Needed for property calls and direct calls
        return self.bound_to(owner)(*args, **kwargs)
