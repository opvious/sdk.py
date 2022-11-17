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
from typing import Any, Mapping, Optional, Protocol


class Executor(Protocol):
    async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
        pass


_GRAPHQL_ENDPOINT = "/graphql"


_TRACE_HEADER = "opvious-trace"


class ApiError(Exception):
    def __init__(self, status, trace, errors=None, extensions=None):
        msg = f"API call failed with status {status} ({trace})"
        if errors:
            msg += ": " + ", ".join(e["message"] for e in errors)
        super().__init__(msg)
        self.status = status
        self.trace = trace
        self.errors = errors
        self.extensions = extensions


def _extract_response_data(
    status: int, trace: Optional[str], body: Any
) -> Any:
    errors = body.get("errors")
    if status != 200 or errors:
        raise ApiError(status, trace, errors, body.get("extensions"))
    return body["data"]


_BROTLI_QUALITY = 4


_COMPRESSION_THRESHOLD = 2**16
# 64 kiB


def aiohttp_executor(url: str, auth: str) -> Executor:
    import aiohttp
    import brotli

    default_headers = {
        "accept-encoding": "br;q=1.0, gzip;q=0.5, *;q=0.1",
        "authorization": auth,
        "content-type": "application/json",
        "opvious-sdk": "Python (aiohttp)",
    }

    class AiohttpExecutor:
        """`aiohttp`-powered GraphQL executor"""

        async def execute(
            self, query: str, variables: Mapping[str, Any]
        ) -> Any:
            data = json.dumps({"query": query, "variables": variables})
            if len(data) > _COMPRESSION_THRESHOLD:
                headers = {"content-encoding": "br"}
                headers.update(default_headers)
                data = brotli.compress(
                    data.encode("utf8"),
                    mode=brotli.MODE_TEXT,
                    quality=_BROTLI_QUALITY,
                )
            else:
                headers = default_headers
            async with aiohttp.ClientSession(headers=headers) as session:
                endpoint = url + _GRAPHQL_ENDPOINT
                async with session.post(endpoint, data=data) as res:
                    trace = res.headers.get(_TRACE_HEADER)
                    body = await res.json()
                    return _extract_response_data(res.status, trace, body)

    return AiohttpExecutor()
