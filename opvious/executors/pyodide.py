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

import contextlib
import json
import logging
from pyodide.http import pyfetch  # type: ignore
from typing import Any, AsyncContextManager, AsyncIterator, Optional
import urllib.parse

from .common import (
    CONTENT_TYPE_HEADER,
    Executor,
    ExecutorResult,
    JsonExecutorResult,
    Headers,
    PlainTextExecutorResult,
    TRACE_HEADER,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


class PyodideExecutor(Executor):
    """`pyodide`-powered executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        super().__init__("pyodide", api_url, authorization)

    def _fetch_result(
        self,
        path: str,
        method: str = "GET",
        headers: Optional[Headers] = None,
        json_body: Optional[Any] = None,
    ) -> AsyncContextManager[ExecutorResult]:
        all_headers = self._headers.copy()
        if headers:
            all_headers.update(headers)
        if json_body:
            all_headers["content-type"] = "application/json"
            data = json.dumps(json_body)
        else:
            data = None
        return _execution(
            url=urllib.parse.urljoin(self._api_url, path),
            method=method,
            headers=all_headers,
            data=data,
        )


@contextlib.asynccontextmanager
async def _execution(
    url: str, method: str, headers: Headers, data: Any
) -> AsyncIterator[ExecutorResult]:
    _logger.debug("Sending API request... [size=%s]", len(data) if data else 0)
    # TODO: Raise any errors as ApiError
    res = await pyfetch(
        url=url,
        method=method,
        headers=headers,
        body=data,
    )
    status = res.status
    headers = res.js_response.headers
    trace = headers.get(TRACE_HEADER)
    ctype = headers.get(CONTENT_TYPE_HEADER)
    if JsonExecutorResult.is_eligible(ctype):
        text = await res.js_response.text()
        yield JsonExecutorResult(status=status, trace=trace, text=text)
    elif PlainTextExecutorResult.is_eligible(ctype):
        text = await res.js_response.text()
        yield PlainTextExecutorResult(status=status, trace=trace, reader=text)
    else:
        # TODO: Support streaming responses.
        raise unsupported_content_type_error(
            content_type=ctype,
            trace=trace,
        )
