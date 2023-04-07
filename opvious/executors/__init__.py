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

import sys
from typing import Optional

from .common import (
    ApiError,
    Executor,
    ExecutorResult,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
)


__all__ = [
    "default_executor",
    "ApiError",
    "Executor",
    "ExecutorResult",
    "JsonExecutorResult",
    "JsonSeqExecutorResult",
    "PlainTextExecutorResult",
]


def aiohttp_executor(
    api_url: str, authorization: Optional[str] = None
) -> Executor:
    from .aiohttp import AiohttpExecutor

    return AiohttpExecutor(api_url, authorization)


def pyodide_executor(
    api_url: str, authorization: Optional[str] = None
) -> Executor:
    from .pyodide import PyodideExecutor

    return PyodideExecutor(api_url=api_url, authorization=authorization)


def urllib_executor(
    api_url: str, authorization: Optional[str] = None
) -> Executor:
    from .urllib import UrllibExecutor

    return UrllibExecutor(api_url=api_url, authorization=authorization)


def _is_using_pyodide():
    # https://pyodide.org/en/stable/usage/faq.html#how-to-detect-that-code-is-run-with-pyodide
    return "pyodide" in sys.modules


def default_executor(
    api_url: str, authorization: Optional[str] = None
) -> Executor:
    """Infers the best executor for the current environment"""
    if _is_using_pyodide():
        return pyodide_executor(api_url=api_url, authorization=authorization)
    try:
        return aiohttp_executor(api_url=api_url, authorization=authorization)
    except ImportError:
        return urllib_executor(api_url=api_url, authorization=authorization)
