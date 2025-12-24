from collections.abc import Callable, Iterable
from datetime import datetime
import functools
from importlib import metadata
import math
from typing import Any, Literal
import urllib.parse
import weakref


try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""


del metadata


type Uuid = str


def if_present[V, R](arg: V | None, fn: Callable[[V], R]) -> R | None:
    return None if arg is None else fn(arg)


# Formatting


type Label = str


def format_percent(val: float | Literal["Infinity"]) -> str:
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


type Json = Any


type ExtendedFloat = float | str


def encode_extended_float(val: ExtendedFloat) -> Json:
    if val == math.inf:
        return "Infinity"
    elif val == -math.inf:
        return "-Infinity"
    return val


def decode_extended_float(val: ExtendedFloat) -> Json:
    match val:
        case "Infinity":
            return math.inf
        case "-Infinity":
            return -math.inf
        case _:
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


type Annotation = str | tuple[str, str]


def encode_annotations(annots: list[Annotation]) -> Json:
    return [
        json_dict(key=annot)
        if isinstance(annot, str)
        else json_dict(key=annot[0], value=str(annot[1]))
        for annot in annots
    ]


def decode_annotations(elems: Json) -> list[Annotation]:
    return [
        elem["key"]
        if elem.get("value") is None
        else (elem["key"], elem["value"])
        for elem in elems
    ]


def decode_datetime(iso: str) -> datetime:
    """Parses a datetime from an ISO-formatted string"""
    return datetime.fromisoformat(iso)


# Async


async def gather(*futures: Any) -> list[Any]:
    """Compatibility shim for asyncio.gather

    It is useful to work in environments which do not support asyncio.
    """
    try:
        import asyncio
    except ImportError:
        ret: list[Any] = []
        for future in futures:
            ret.append(await future)
        return ret
    else:
        return await asyncio.gather(*futures)


# Decorator utilities


_lambda = lambda: 0  # noqa


def _is_lambda(fn: Callable[..., Any]) -> bool:
    return getattr(fn, "__name__", None) == _lambda.__name__


def capturing_instance(wrapper: Callable[..., Any]) -> Any:
    def wrap(fn: Callable[..., Any]) -> Bindable:
        return Bindable(fn, wrapper)

    return wrap


def with_instance(consumer: Callable[..., Any]) -> Any:
    def wrap(fn: Callable[..., Any]) -> Bindable:
        return Bindable(fn, consumer, lazy=True)

    return wrap


def method_decorator[F: Callable[..., Any]](
    require_call: bool = False,
) -> Callable[[F], Any]:
    """Transforms a decorator into a method-friendly equivalent"""

    def wrap_decorator(decorator: F) -> Any:
        @functools.wraps(decorator)
        def wrapped_decorator(*args: Any, **kwargs: Any) -> Any:
            arg = args[0] if args else None
            if callable(arg):
                if _is_lambda(arg):
                    # Lazy decorator constructor
                    if len(args) > 1 or kwargs:
                        raise Exception("Unexpected tail arguments")

                    def wrap_method(meth: Any) -> Bindable:
                        return Bindable(
                            meth, lambda self: arg(decorator, self), lazy=True
                        )

                    return wrap_method
                elif not require_call and len(args) == 1 and not kwargs:
                    # No argument decorator
                    return Bindable(arg, decorator())
            else:
                # Standard decorator creation

                def wrap_method(meth: Any) -> Bindable:
                    return Bindable(meth, decorator(*args, **kwargs))

                return wrap_method

        return wrapped_decorator

    return wrap_decorator


class Bindable:
    """Container for decorated instance attributes"""

    def __init__(
        self,
        body: Callable[..., Any],
        wrap: Callable[..., Any],
        lazy: bool = False,
    ) -> None:
        self._body = body
        self._wrap = wrap
        self._lazy = lazy
        self._bindings: Any = weakref.WeakKeyDictionary()
        self.__doc__ = self._body.__doc__

    def _apply(self, owner: Any, bind: bool = True) -> Any:
        wrap = self._wrap(owner) if self._lazy else self._wrap
        body = functools.partial(self._body, owner) if bind else self._body
        wrapper = wrap(body)
        if wrapper is not None:
            functools.update_wrapper(wrapper, self._body)
        return wrapper

    def bound_to(self, owner: Any) -> Any:
        binding = self._bindings.get(owner)
        if not binding:
            binding = self._apply(owner)
            while isinstance(binding, Bindable):
                binding = binding._apply(owner, False)  # noqa
            self._bindings[owner] = binding
        return binding

    def __get__(self, owner: Any, _objtype: Any = None) -> Any:
        if owner is None:  # Accessed via the class
            return self._body
        return self.bound_to(owner)

    def __call__(self, owner: Any, *args: Any, **kwargs: Any) -> Any:
        # Needed for property calls and direct calls
        return self.bound_to(owner)(*args, **kwargs)
