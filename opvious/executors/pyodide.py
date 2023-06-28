import contextlib
import logging
from pyodide.http import pyfetch  # type: ignore
from typing import AsyncIterator, Optional

from .common import (
    CONTENT_TYPE_HEADER,
    Executor,
    ExecutorResult,
    JsonExecutorResult,
    Headers,
    PlainTextExecutorResult,
    TRACE_HEADER,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


class PyodideExecutor(Executor):
    """`pyodide`-powered executor"""

    def __init__(self, endpoint: str, authorization: Optional[str] = None):
        super().__init__(
            variant="pyodide", endpoint=endpoint, authorization=authorization
        )

    @contextlib.asynccontextmanager
    async def _send(
        self, url: str, method: str, headers: Headers, body: Optional[bytes]
    ) -> AsyncIterator[ExecutorResult]:
        res = await pyfetch(
            url=url,
            method=method,
            headers=headers,
            body=body,
        )
        status = res.status
        headers = res.js_response.headers
        trace = headers.get(TRACE_HEADER)
        ctype = headers.get(CONTENT_TYPE_HEADER)
        if JsonExecutorResult.is_eligible(ctype):
            text = await res.js_response.text()
            yield JsonExecutorResult(status=status, trace=trace, text=text)
        elif PlainTextExecutorResult.is_eligible(ctype):
            text = await res.js_response.text()
            yield PlainTextExecutorResult(
                status=status, trace=trace, reader=text
            )
        else:
            raise unsupported_content_type_error(
                content_type=ctype,
                trace=trace,
            )
