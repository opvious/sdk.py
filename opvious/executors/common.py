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

import asyncio
import dataclasses
import json
import logging
from typing import Any, AsyncIterator, Mapping, Optional, Protocol


_logger = logging.getLogger(__name__)


GRAPHQL_ENDPOINT = "/graphql"


TRACE_HEADER = "opvious-trace"


CONTENT_TYPE_HEADER = "content-type"


Headers = Mapping[str, str]


class ApiError(Exception):
    """Local representation of an error returned by the API"""

    def __init__(
        self,
        status: int,
        trace: Optional[str] = None,
        data: Optional[Any] = None,
    ):
        message = f"API call failed with status {status}"
        if trace:
            message += f" ({trace})"
        if data:
            message += f": {data}"
        super().__init__(message)
        self.status = status
        self.trace = trace
        self.data = data


def unexpected_response_error(
    message: str, trace: Optional[str] = None
) -> Exception:
    return Exception(
        "Unexpected response"
        + (f" ({trace})" if trace else "")
        + (f": {message}" if message else "")
    )


def unsupported_content_type_error(
    content_type: str, trace: Optional[str] = None
) -> Exception:
    return unexpected_response_error(
        message=f"unsupported content-type: {content_type}",
        trace=trace,
    )


@dataclasses.dataclass
class ExecutorResult:
    status: int
    trace: Optional[str]

    def __post_init__(self):
        _logger.debug(
            "Got API response. [status=%s, trace=%s]", self.status, self.trace
        )

    def _assert_status(self, status: int, text: Optional[str] = None) -> None:
        if self.status != status:
            raise ApiError(status=self.status, trace=self.trace, data=text)

    @classmethod
    def is_eligible(cls, ctype: Optional[str]) -> bool:
        return ctype and ctype.split(";")[0] == cls.content_type


@dataclasses.dataclass
class JsonExecutorResult(ExecutorResult):
    """Unary JSON execution result"""

    text: str = dataclasses.field(repr=False)
    content_type = "application/json"

    def json_data(self, status: int = 200) -> Any:
        self._assert_status(status, self.text)
        return json.loads(self.text)


@dataclasses.dataclass
class JsonSeqExecutorResult(ExecutorResult):
    """Streaming JSON execution result"""

    reader: asyncio.StreamReader = dataclasses.field(repr=False)
    content_type = "application/json-seq"

    async def json_seq_data(self) -> AsyncIterator[Any]:
        self._assert_status(200)
        if hasattr(self.reader, "__aiter__"):
            async for line in self.reader:
                yield _json_seq_item(line)
        else:
            # For synchronous executors
            for line in self.reader:
                yield _json_seq_item(line)


RECORD_SEPARATOR = b"\x1e"


def _json_seq_item(line: str) -> Any:
    # TODO: Robust error checking.
    data = line[1:] if line.startswith(RECORD_SEPARATOR) else line
    return json.loads(data)


Execution = AsyncIterator[ExecutorResult]


class Executor(Protocol):
    def execute(
        self,
        path: str,
        method: str = "GET",
        headers: Optional[Headers] = None,
        json_body: Optional[str] = None,
    ) -> Execution:
        pass


async def execute_graphql_query(
    executor: Executor,
    query: str,
    variables: Optional[Mapping[str, Any]] = None,
) -> Any:
    async with executor.execute(
        path="/graphql",
        method="POST",
        json_body={"query": query, "variables": variables or {}},
    ) as result:
        data = result.json_data()
    if data.get("errors"):
        raise ApiError(status=result.status, trace=result.trace, data=data)
    return data["data"]


def default_headers(client: str) -> Headers:
    return {
        "accept": "application/json;q=1, text/*;q=0.1",
        "opvious-client": f"Python SDK ({client})",
    }
