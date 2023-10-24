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


AUTHORIZATION_HEADER = "authorization"


CONTENT_TYPE_HEADER = "content-type"


Headers = dict[str, str]


@dataclasses.dataclass(init=False)
class ExecutorError(Exception):
    """Local representation of an error during an executor's request"""

    status: Optional[int]
    trace: Optional[str]
    reason: Optional[str]

    def __init__(
        self,
        status: Optional[int] = None,
        trace: Optional[str] = None,
        reason: Optional[Any] = None,
    ) -> None:
        message = "Request errored"
        if status and status >= 400:
            message += f" with status {status}"
        if trace:
            message += f" ({trace})"
        if reason:
            message += f": {reason}"
        super().__init__(message)
        self.status = status
        self.trace = trace
        self.reason = reason


def unsupported_content_type_error(
    text: str, content_type: Optional[str], trace: Optional[str] = None
) -> Exception:
    reason = f"unsupported content-type ({content_type or '<none>'}): {text}"
    return ExecutorError(trace=trace, reason=reason)


@dataclasses.dataclass
class ExecutorResult:
    """Request execution result"""

    status: int
    """Response HTTP status code"""

    trace: Optional[str]
    """Request trace ID"""

    def __post_init__(self):
        _logger.debug(
            "Got executor result. [status=%s, trace=%s]",
            self.status,
            self.trace,
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
        raise ExecutorError(status=self.status, trace=self.trace, reason=text)

    @classmethod
    def is_eligible(cls, ctype: Optional[str]) -> bool:
        return bool(ctype and ctype.split(";")[0] == cls.content_type)


@dataclasses.dataclass
class PlainTextExecutorResult(ExecutorResult):
    """Plain text execution result"""

    content_type = "text/plain"
    reader: Any = dataclasses.field(repr=False)

    async def text(self, assert_status: Optional[int] = None) -> str:
        lines = []
        async for line in self.lines(assert_status=assert_status):
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
    """Generic HTTP request executor"""

    def __init__(
        self,
        variant: str,
        endpoint: str,
        authorization: Optional[str] = None,
        supports_streaming=False,
    ):
        self._variant = variant
        self._endpoint = endpoint
        self._root_headers = _default_headers(variant)
        if authorization:
            self._root_headers[AUTHORIZATION_HEADER] = authorization
        self.supports_streaming = supports_streaming
        _logger.debug("Instantiated %s executor. [url=%s]", variant, endpoint)

    @property
    def endpoint(self) -> str:
        """The executor's root endpoint, used for all relative URLs"""
        return self._endpoint

    @property
    def authenticated(self):
        """Whether requests use an API authorization header"""
        return AUTHORIZATION_HEADER in self._root_headers

    def _send(
        self, url: str, method: str, headers: Headers, body: Optional[bytes]
    ) -> AsyncContextManager[ExecutorResult]:
        raise NotImplementedError()

    @contextlib.asynccontextmanager
    async def execute(
        self,
        result_type: Type[ExpectedExecutorResult],
        url: str,
        method: str = "GET",
        headers: Optional[Headers] = None,
        json_data: Optional[Any] = None,
    ) -> AsyncIterator[ExpectedExecutorResult]:
        """Send a request"""
        full_url = urllib.parse.urljoin(self.endpoint, url)
        if full_url.startswith(self.endpoint):
            all_headers = self._root_headers.copy()
        else:
            all_headers = {}
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
            "Sending %s executor request... [size=%s, url=%s]",
            self._variant,
            len(body) if body else 0,
            full_url,
        )
        async with self._send(
            url=full_url,
            method=method,
            headers=all_headers,
            body=body,
        ) as result:
            if not isinstance(result, result_type):
                if isinstance(result, PlainTextExecutorResult):
                    reason = await result.text()
                elif isinstance(result, JsonExecutorResult):
                    reason = result.text
                else:
                    reason = None
                raise ExecutorError(
                    status=result.status, trace=result.trace, reason=reason
                )
            yield result

    async def execute_graphql_query(
        self,
        query: str,
        variables: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Send a GraphQL API request"""
        async with self.execute(
            result_type=JsonExecutorResult,
            url="/graphql",
            method="POST",
            json_data={"query": query, "variables": variables or {}},
        ) as result:
            data = result.json_data()
        if data.get("errors"):
            raise ExecutorError(
                status=result.status,
                trace=result.trace,
                reason=json.dumps(data),
            )
        return data["data"]


def _default_headers(client: str) -> Headers:
    return {
        "opvious-client": f"Python SDK v{__version__} ({client})",
    }
