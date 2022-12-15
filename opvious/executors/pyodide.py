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
from pyodide.http import pyfetch
from typing import Optional

from .common import extract_api_data, GRAPHQL_ENDPOINT, TRACE_HEADER


_DEFAULT_HEADERS = {
    "content-type": "application/json",
    "opvious-client": "Python SDK (pyodide)",
}


class PyodideExecutor:
    """`pyodide`-powered executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._endpoint = api_url + GRAPHQL_ENDPOINT
        self._headers = _DEFAULT_HEADERS.copy()
        if authorization:
            self._headers["authorization"] = authorization

    async def execute(self, query, variables):
        res = await pyfetch(
            self._endpoint,
            method="POST",
            headers=self._headers,
            body=json.dumps({"query": query, "variables": variables}),
        )
        body = await res.js_response.text()
        return extract_api_data(
            status=res.status,
            trace=res.js_response.headers.get(TRACE_HEADER),
            body=body,
        )
