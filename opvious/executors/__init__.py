import base64
import io
import pandas as pd
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
    "authorization_header",
    "aiohttp_executor",
    "default_executor",
    "pyodide_executor",
    "urllib_executor",
    "default_executor",
    "Executor",
    "ExecutorError",
    "ExecutorResult",
    "JsonExecutorResult",
    "JsonSeqExecutorResult",
    "PlainTextExecutorResult",
    "fetch_csv",
    "fetch_text",
]


def aiohttp_executor(
    endpoint: str, authorization: Optional[str] = None
) -> Executor:
    """Returns an `aiohttp`-powered executor"""
    from .aiohttp import AiohttpExecutor

    return AiohttpExecutor(endpoint, authorization)


def pyodide_executor(
    endpoint: str, authorization: Optional[str] = None
) -> Executor:
    """Returns a Pyodide-compatible executor"""
    from .pyodide import PyodideExecutor

    return PyodideExecutor(endpoint=endpoint, authorization=authorization)


def urllib_executor(
    endpoint: str, authorization: Optional[str] = None
) -> Executor:
    """Returns a native `urllib` executor"""
    from .urllib import UrllibExecutor

    return UrllibExecutor(endpoint=endpoint, authorization=authorization)


def _is_using_pyodide():
    # https://pyodide.org/en/stable/usage/faq.html#how-to-detect-that-code-is-run-with-pyodide
    return "pyodide" in sys.modules


def default_executor(
    endpoint: str, authorization: Optional[str] = None
) -> Executor:
    """Infers the best executor for the current environment"""
    if _is_using_pyodide():
        return pyodide_executor(endpoint=endpoint, authorization=authorization)
    try:
        return aiohttp_executor(endpoint=endpoint, authorization=authorization)
    except ImportError:
        return urllib_executor(endpoint=endpoint, authorization=authorization)


def authorization_header(token: str) -> str:
    """Generates a suitable authorization header from a token"""
    if " " in token:
        return token
    if ":" in token:
        value = base64.b64encode(token.encode("utf8")).decode("utf8")
        return f"Basic {value}"
    return f"Bearer {token}"


async def fetch_text(url: str) -> str:
    """Fetches a URL's contents as plain text

    Args:
        url: The URL to fetch (via GET)
    """
    # We specify an invalid endpoint URL so that default headers are never
    # included in the request. Otherwise it causes certain requests to fail
    # (e.g. when fetching gists).
    executor = default_executor("ignored://")
    async with executor.execute(PlainTextExecutorResult, url) as res:
        return await res.text()


async def fetch_csv(url: str, **kwargs) -> pd.DataFrame:
    """Fetches a CSV from a URL

    This convenience method is provided as a portable way to fetch data from
    notebooks. Using the default `pandas.read_csv` does not support `https://`
    URLs from JupyterLite (Pyodide).

    Args:
        url: The URL to fetch (via GET)
        **kwargs: Keyword arguments forwarded to `pandas.read_csv`
    """
    text = await fetch_text(url)
    return pd.read_csv(io.StringIO(text), **kwargs)
