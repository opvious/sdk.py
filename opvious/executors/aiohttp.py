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
import json
import logging
import urllib.parse
from typing import Any, Optional

from .common import default_headers, ExecutorResult, TRACE_HEADER


_logger = logging.getLogger(__name__)


_BROTLI_QUALITY = 4


_COMPRESSION_THRESHOLD = 2**16


class AiohttpExecutor:
    """`aiohttp`-powered GraphQL executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._api_url = api_url
        self._headers = default_headers("aiohttp")
        self._headers["accept-encoding"] = "br;q=1.0, gzip;q=0.5, *;q=0.1"
        if authorization:
            self._headers["authorization"] = authorization
        _logger.info("Instantiated `aiohttp` executor.")

    async def execute(
        self, path: str, method: str = "GET", body: Optional[Any] = None
    ) -> ExecutorResult:
        headers = self._headers.copy()
        if body:
            headers["content-type"] = "application/json"
            data = json.dumps(body)
            if len(data) > _COMPRESSION_THRESHOLD:
                headers["content-encoding"] = "br"
                data = brotli.compress(
                    data.encode("utf8"),
                    mode=brotli.MODE_TEXT,
                    quality=_BROTLI_QUALITY,
                )
        else:
            data = None
        url = urllib.parse.urljoin(self._api_url, path)
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.request(
                method=method, url=url, data=data
            ) as res:
                trace = res.headers.get(TRACE_HEADER)
                body = await res.text()
                return ExecutorResult(
                    status=res.status, body=body, trace=trace
                )
