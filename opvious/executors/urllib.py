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
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .common import (
    CONTENT_TYPE_HEADER,
    default_headers,
    Headers,
    Execution,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    TRACE_HEADER,
    unexpected_response_error,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


class UrllibExecutor:
    """
    `urllib`-powered GraphQL executor, used as fallback. When possible, prefer
    using the `aiohttp` equivalent.
    """

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._api_url = api_url
        self._headers = default_headers("urllib")
        if authorization:
            self._headers["authorization"] = authorization
        _logger.debug(
            "Instantiated executor. [name=%s, url=%s]",
            self.__class__.__name__,
            api_url,
        )

    def execute(
        self,
        path: str,
        method: str = "GET",
        headers: Optional[Headers] = None,
        json_body: Optional[Any] = None,
    ) -> Execution:
        all_headers = self._headers.copy()
        if headers:
            all_headers.update(headers)
        if json_body:
            all_headers["content-type"] = "application/json"
            data = json.dumps(json_body).encode("utf8")
        else:
            data = None
        return _execution(
            url=urllib.parse.urljoin(self._api_url, path),
            headers=all_headers,
            method=method,
            data=data,
        )


@contextlib.asynccontextmanager
async def _execution(
    url: str, method: str, headers: Headers, data: Any
) -> Execution:
    _logger.debug("Sending API request... [size=%s]", len(data) if data else 0)
    req = urllib.request.Request(
        url=url,
        headers=headers,
        method=method,
        data=data,
    )
    try:
        res = urllib.request.urlopen(req)
    except urllib.error.HTTPError as err:
        res = err
    except Exception as err:
        trace = err.headers.get(TRACE_HEADER)
        raise unexpected_response_error(message=err.reason, trace=trace)
    status = res.status
    trace = res.getheader(TRACE_HEADER)
    ctype = res.getheader(CONTENT_TYPE_HEADER)
    if JsonExecutorResult.is_eligible(ctype):
        text = res.read().decode("utf8")
        yield JsonExecutorResult(status=status, trace=trace, text=text)
    elif JsonSeqExecutorResult.is_eligible(ctype):
        # TODO: Check compatible
        yield JsonSeqExecutorResult(status=status, trace=trace, reader=res)
    else:
        raise unsupported_content_type_error(
            content_type=ctype,
            trace=trace,
        )
