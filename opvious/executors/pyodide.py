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
from pyodide.http import pyfetch
from typing import Any, Optional
import urllib.parse

from .common import default_headers, ExecutorResult, TRACE_HEADER


_logger = logging.getLogger(__name__)


class PyodideExecutor:
    """`pyodide`-powered executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._api_url = api_url
        self._headers = default_headers("pyodide")
        if authorization:
            self._headers["authorization"] = authorization
        _logger.info("Instantiated Pyodide executor.")

    async def execute(
        self, path: str, method: str = "GET", body: Optional[Any] = None
    ) -> ExecutorResult:
        headers = self._headers.copy()
        if body:
            headers["content-type"] = "application/json"
            data = json.dumps(body)
        else:
            data = None
        res = await pyfetch(
            urllib.parse.urljoin(self._api_url, path),
            method=method,
            headers=self._headers,
            body=data,
        )
        text = await res.js_response.text()
        return ExecutorResult(
            status=res.status,
            trace=res.js_response.headers.get(TRACE_HEADER),
            body=text,
        )
