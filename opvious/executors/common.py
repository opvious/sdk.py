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

import dataclasses
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
    def from_graphql(
        cls, status, trace, errors, extensions=None
    ) -> "ApiError":
        msg = f"API call failed with status {status} ({trace})"
        if errors:
            msg += ": " + ", ".join(e["message"] for e in errors)
        return ApiError(status, msg, trace, errors, extensions)


@dataclasses.dataclass
class ExecutorResult:
    status: int
    body: str
    trace: Optional[str] = None

    def json_data(self, status: int = 200) -> Any:
        if self.status != status:
            raise ApiError(
                status=self.status,
                message=(
                    f"Unexpected {self.status} API response ({self.trace}): "
                    + self.body
                ),
                trace=self.trace,
            )
        return json.loads(self.body)


class Executor(Protocol):
    async def execute(
        self, path: str, method: str = "GET", body: Optional[str] = None
    ) -> ExecutorResult:
        pass


async def execute_graphql_query(
    executor: Executor,
    query: str,
    variables: Optional[Mapping[str, Any]] = None,
) -> Any:
    result = await executor.execute(
        path="/graphql",
        method="POST",
        body={"query": query, "variables": variables or {}},
    )
    data = result.json_data()
    errors = data.get("errors")
    if errors:
        extensions = data.get("extensions")
        raise ApiError.from_graphql(
            status=result.status,
            trace=result.trace,
            errors=errors,
            extensions=extensions,
        )
    return data["data"]


def default_headers(client: str) -> Mapping[str, str]:
    return {
        "accept": "application/json;q=1, text/*;q=0.1",
        "opvious-client": f"Python SDK ({client})",
    }
