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

import contextlib
import dataclasses
import json
import logging
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Type,
    Mapping,
    Optional,
    TypeVar,
)
import urllib.parse

from ..common import __version__


_logger = logging.getLogger(__name__)


TRACE_HEADER = "opvious-trace"


CONTENT_TYPE_HEADER = "content-type"


Headers = dict[str, str]


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
    content_type: Optional[str], trace: Optional[str] = None
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

    @property
    def accept(self) -> str:
        raise NotImplementedError()

    @property
    def content_type(self) -> str:
        raise NotImplementedError()

    def _assert_status(self, status: int, text: Optional[str] = None) -> None:
        if self.status == status:
            return
        raise ApiError(status=self.status, trace=self.trace, data=text)

    @classmethod
    def is_eligible(cls, ctype: Optional[str]) -> bool:
        return bool(ctype and ctype.split(";")[0] == cls.content_type)


@dataclasses.dataclass
class PlainTextExecutorResult(ExecutorResult):
    """Plain text execution result"""

    content_type = "text/plain"
    reader: Any = dataclasses.field(repr=False)

    async def text(self) -> str:
        lines = []
        async for line in self.lines(assert_status=None):
            lines.append(line)
        return "".join(lines)

    async def lines(
        self, assert_status: Optional[int] = 200
    ) -> AsyncIterator[str]:
        if assert_status:
            self._assert_status(assert_status)

        # Non-streaming
        if isinstance(self.reader, str):
            for line in self.reader.splitlines(keepends=True):
                yield line
            return

        # Streaming
        splitter = _LineSplitter()
        if hasattr(self.reader, "__aiter__"):
            async for chunk in self.reader:
                for line in splitter.push(chunk):
                    yield line
        elif hasattr(self.reader, "__iter__"):
            for chunk in self.reader:
                for line in splitter.push(chunk):
                    yield line
        else:
            raise Exception(f"Non-iterable reader: {self.reader}")
        yield splitter.flush()


class _LineSplitter:
    def __init__(self):
        self._buffer = ""

    def push(self, chunk: bytes) -> list[str]:
        lines = chunk.decode("utf8").splitlines(keepends=True)
        if self._buffer:
            lines[0] = self._buffer + lines[0]
        if lines[-1].endswith("\n"):
            self._buffer = ""
        else:
            self._buffer = lines.pop()
        return lines

    def flush(self) -> str:
        buf = self._buffer
        self._buffer = ""
        return buf


@dataclasses.dataclass
class JsonExecutorResult(ExecutorResult):
    """Unary JSON execution result"""

    content_type = "application/json"
    text: str = dataclasses.field(repr=False)

    def json_data(self, status: int = 200) -> Any:
        self._assert_status(status, self.text)
        return json.loads(self.text)


@dataclasses.dataclass
class JsonSeqExecutorResult(ExecutorResult):
    """Streaming JSON execution result"""

    content_type = "application/json-seq"
    reader: Any = dataclasses.field(repr=False)

    async def json_seq_data(self) -> AsyncIterator[Any]:
        self._assert_status(200)
        if hasattr(self.reader, "__aiter__"):
            async for line in self.reader:
                yield _json_seq_item(line)
        elif hasattr(self.reader, "__iter__"):
            # For synchronous executors
            for line in self.reader:
                yield _json_seq_item(line)
        else:
            raise Exception(f"Non-iterable reader: {self.reader}")


RECORD_SEPARATOR = b"\x1e"


def _json_seq_item(line: bytes) -> Any:
    # TODO: Robust error checking.
    data = line[1:] if line.startswith(RECORD_SEPARATOR) else line
    return json.loads(data)


ExpectedExecutorResult = TypeVar(
    "ExpectedExecutorResult", bound=ExecutorResult
)


class Executor:
    def __init__(
        self, variant: str, api_url: str, authorization: Optional[str] = None
    ):
        self._api_url = api_url
        self._headers = _default_headers(variant)
        if authorization:
            self._headers["authorization"] = authorization
        _logger.debug("Instantiated %s executor. [url=%s]", variant, api_url)

    def _send(
        self, url: str, method: str, headers: Headers, body: Optional[bytes]
    ) -> AsyncContextManager[ExecutorResult]:
        raise NotImplementedError()

    @contextlib.asynccontextmanager
    async def execute(
        self,
        result_type: Type[ExpectedExecutorResult],
        path: str,
        method: str = "GET",
        headers: Optional[Headers] = None,
        json_data: Optional[Any] = None,
    ) -> AsyncIterator[ExpectedExecutorResult]:
        all_headers = self._headers.copy()
        if headers:
            all_headers.update(headers)

        if result_type == JsonExecutorResult:
            accept = "application/json;q=1, text/plain;q=0.1"
        elif result_type == JsonSeqExecutorResult:
            accept = "application/json-seq;q=1, text/plain;q=0.1"
        elif result_type == PlainTextExecutorResult:
            accept = "text/plain"
        else:
            raise Exception(f"Unsupported result type: {result_type}")
        all_headers["accept"] = accept

        if json_data:
            all_headers["content-type"] = "application/json"
            body = json.dumps(json_data).encode("utf8")
        else:
            body = None

        _logger.debug(
            "Sending API request... [size=%s]", len(body) if body else 0
        )
        async with self._send(
            url=urllib.parse.urljoin(self._api_url, path),
            method=method,
            headers=all_headers,
            body=body,
        ) as result:
            if not isinstance(result, result_type):
                if isinstance(self, PlainTextExecutorResult):
                    text = await self.text()
                else:
                    text = None
                raise ApiError(
                    status=result.status, trace=result.trace, data=text
                )
            yield result

    async def execute_graphql_query(
        self,
        query: str,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        async with self.execute(
            result_type=JsonExecutorResult,
            path="/graphql",
            method="POST",
            json_data={"query": query, "variables": variables or {}},
        ) as result:
            data = result.json_data()
        if data.get("errors"):
            raise ApiError(status=result.status, trace=result.trace, data=data)
        return data["data"]


def _default_headers(client: str) -> Headers:
    return {
        "opvious-client": f"Python SDK v{__version__} ({client})",
    }
