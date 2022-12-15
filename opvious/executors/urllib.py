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
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

from .common import ApiError, extract_api_data, GRAPHQL_ENDPOINT, TRACE_HEADER


_DEFAULT_HEADERS = {
    "content-type": "application/json",
    "opvious-sdk": "Python (urllib)",
}


class UrllibExecutor:
    """
    `urllib`-powered GraphQL executor, used as fallback. When possible, prefer
    using the `aiohttp` equivalent.
    """

    def __init__(self, api_url: str, authorization: Optional[str] = None):
        self._endpoint = api_url + GRAPHQL_ENDPOINT
        self._headers = _DEFAULT_HEADERS.copy()
        if authorization:
            self._headers["authorization"] = authorization

    async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
        body = json.dumps({"query": query, "variables": variables})
        req = urllib.request.Request(
            url=self._endpoint,
            headers=self._headers,
            data=body.encode("utf8"),
            method="POST",
        )
        try:
            res = urllib.request.urlopen(req)
        except urllib.error.HTTPError as err:
            raise ApiError(
                status=err.code,
                message=err.reason,
                trace=err.headers.get(TRACE_HEADER),
            )
        return extract_api_data(
            status=res.status,
            trace=res.getheader(TRACE_HEADER),
            body=res.read().decode("utf8"),
        )
