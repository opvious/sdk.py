import sys
from typing import Optional

from .common import (
    Executor,
    ExecutorError,
    ExecutorResult,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
)


__all__ = [
    "default_executor",
    "Executor",
    "ExecutorError",
    "ExecutorResult",
    "JsonExecutorResult",
    "JsonSeqExecutorResult",
    "PlainTextExecutorResult",
]


def aiohttp_executor(
    root_url: str, authorization: Optional[str] = None
) -> Executor:
    """Returns an `aiohttp`-powered executor"""
    from .aiohttp import AiohttpExecutor

    return AiohttpExecutor(root_url, authorization)


def pyodide_executor(
    root_url: str, authorization: Optional[str] = None
) -> Executor:
    """Returns a Pyodide-compatible executor"""
    from .pyodide import PyodideExecutor

    return PyodideExecutor(root_url=root_url, authorization=authorization)


def urllib_executor(
    root_url: str, authorization: Optional[str] = None
) -> Executor:
    """Returns a native `urllib` executor"""
    from .urllib import UrllibExecutor

    return UrllibExecutor(root_url=root_url, authorization=authorization)


def _is_using_pyodide():
    # https://pyodide.org/en/stable/usage/faq.html#how-to-detect-that-code-is-run-with-pyodide
    return "pyodide" in sys.modules


def default_executor(
    root_url: str, authorization: Optional[str] = None
) -> Executor:
    """Infers the best executor for the current environment"""
    if _is_using_pyodide():
        return pyodide_executor(root_url=root_url, authorization=authorization)
    try:
        return aiohttp_executor(root_url=root_url, authorization=authorization)
    except ImportError:
        return urllib_executor(root_url=root_url, authorization=authorization)
