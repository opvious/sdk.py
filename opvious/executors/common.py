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


GRAPHQL_ENDPOINT = "/graphql"


TRACE_HEADER = "opvious-trace"


class ApiError(Exception):
    def __init__(
        self, status, message, trace=None, errors=None, extensions=None
    ):
        super().__init__(message)
        self.status = status
        self.trace = trace
        self.errors = errors
        self.extensions = extensions

    @classmethod
    def from_graphql(cls, status, trace, errors, extensions=None):
        msg = f"API call failed with status {status} ({trace})"
        if errors:
            msg += ": " + ", ".join(e["message"] for e in errors)
        return ApiError(status, msg, trace, errors, extensions)


def extract_api_data(status: int, trace: Optional[str], body: str) -> Any:
    try:
        data = json.loads(body)
    except Exception:
        raise ApiError(
            status=status,
            message=f"Unexpected API response ({trace}): ${body}",
            trace=trace,
        )
    errors = data.get("errors")
    if status != 200 or errors:
        extensions = data.get("extensions")
        raise ApiError.from_graphql(status, trace, errors, extensions)
    return data["data"]


class Executor(Protocol):
    async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
        pass
