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

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from .common import ApiError, default_headers, ExecutorResult, TRACE_HEADER


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
        _logger.info("Instantiated `urllib` executor.")

    async def execute(
        self, path: str, method: str = "GET", body: Optional[Any] = None
    ) -> ExecutorResult:
        headers = self._headers.copy()
        if body:
            headers["content-type"] = "application/json"
            data = json.dumps(body).encode("utf8")
        else:
            data = None
        req = urllib.request.Request(
            url=urllib.parse.urljoin(self._api_url, path),
            headers=headers,
            data=data,
            method=method,
        )
        try:
            res = urllib.request.urlopen(req)
        except urllib.error.HTTPError as err:
            raise ApiError(
                status=err.code,
                message=err.reason,
                trace=err.headers.get(TRACE_HEADER),
            )
        return ExecutorResult(
            status=res.status,
            trace=res.getheader(TRACE_HEADER),
            body=res.read().decode("utf8"),
        )
