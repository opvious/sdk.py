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

import aiohttp
import brotli
import contextlib
import json
import logging
import urllib.parse
from typing import Any, Optional

from .common import (
    CONTENT_TYPE_HEADER,
    default_headers,
    Execution,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    Headers,
    TRACE_HEADER,
    unexpected_response_error,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


_BROTLI_QUALITY = 4


_COMPRESSION_THRESHOLD = 2**16


_READ_BUFFER_SIZE = 2**26  # 512MiB


_REQUEST_TIMEOUT_SECONDS = 900  # 15 minutes


class AiohttpExecutor:
    """`aiohttp`-powered GraphQL executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._api_url = api_url
        self._headers = default_headers("aiohttp")
        self._headers["accept-encoding"] = "br;q=1.0, gzip;q=0.5, *;q=0.1"
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
            data = json.dumps(json_body)
            if len(data) > _COMPRESSION_THRESHOLD:
                _logger.debug(
                    "Compressing API request... [size=%s]", len(data)
                )
                all_headers["content-encoding"] = "br"
                data = brotli.compress(
                    data.encode("utf8"),
                    mode=brotli.MODE_TEXT,
                    quality=_BROTLI_QUALITY,
                )
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
) -> Execution:
    _logger.debug("Sending API request... [size=%s]", len(data) if data else 0)
    try:
        async with aiohttp.ClientSession(
            headers=headers,
            read_bufsize=_READ_BUFFER_SIZE,
            timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
        ) as session:
            async with session.request(
                method=method, url=url, data=data
            ) as res:
                status = res.status
                trace = res.headers.get(TRACE_HEADER)
                ctype = res.headers.get(CONTENT_TYPE_HEADER)
                if JsonExecutorResult.is_eligible(ctype):
                    text = await res.text()
                    yield JsonExecutorResult(
                        status=status,
                        trace=trace,
                        text=text,
                    )
                elif JsonSeqExecutorResult.is_eligible(ctype):
                    yield JsonSeqExecutorResult(
                        status=status,
                        trace=trace,
                        reader=res.content,
                    )
                else:
                    raise unsupported_content_type_error(
                        content_type=ctype,
                        trace=trace,
                    )
    except aiohttp.ClientResponseError as err:
        trace = next((v for k, v in err.headers if k == TRACE_HEADER), None)
        raise unexpected_response_error(message=err.message, trace=trace)
