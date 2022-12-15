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
from typing import Any, Mapping, Optional

from .common import extract_api_data, GRAPHQL_ENDPOINT, TRACE_HEADER


_BROTLI_QUALITY = 4


_COMPRESSION_THRESHOLD = 2**16


_DEFAULT_HEADERS = {
    "accept-encoding": "br;q=1.0, gzip;q=0.5, *;q=0.1",
    "content-type": "application/json",
    "opvious-sdk": "Python (aiohttp)",
}


class AiohttpExecutor:
    """`aiohttp`-powered GraphQL executor"""

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._endpoint = api_url + GRAPHQL_ENDPOINT
        self._headers = _DEFAULT_HEADERS.copy()
        if authorization:
            self._headers["authorization"] = authorization

    async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
        headers = self._headers.copy()
        data = json.dumps({"query": query, "variables": variables})
        if len(data) > _COMPRESSION_THRESHOLD:
            headers["content-encoding"] = "br"
            data = brotli.compress(
                data.encode("utf8"),
                mode=brotli.MODE_TEXT,
                quality=_BROTLI_QUALITY,
            )
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(self._endpoint, data=data) as res:
                trace = res.headers.get(TRACE_HEADER)
                body = await res.text()
                return extract_api_data(res.status, trace, body)
