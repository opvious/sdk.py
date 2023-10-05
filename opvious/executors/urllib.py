import contextlib
import logging
import urllib.error
import urllib.request
from typing import AsyncIterator, Optional

from .common import (
    CONTENT_TYPE_HEADER,
    Headers,
    Executor,
    ExecutorResult,
    JsonExecutorResult,
    PlainTextExecutorResult,
    TRACE_HEADER,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


class UrllibExecutor(Executor):
    """
    `urllib`-powered executor, used as fallback. When possible, prefer using
    the `aiohttp` equivalent.
    """

    def __init__(self, endpoint: str, authorization: Optional[str] = None):
        super().__init__(
            variant="urllib", endpoint=endpoint, authorization=authorization
        )

    @contextlib.asynccontextmanager
    async def _send(
        self, url: str, method: str, headers: Headers, body: Optional[bytes]
    ) -> AsyncIterator[ExecutorResult]:
        req = urllib.request.Request(
            url=url,
            headers=headers,
            method=method,
            data=body,
        )
        try:
            res = urllib.request.urlopen(req)
        except urllib.error.HTTPError as err:
            res = err
        status = res.status
        trace = res.getheader(TRACE_HEADER)
        ctype = res.getheader(CONTENT_TYPE_HEADER)
        if JsonExecutorResult.is_eligible(ctype):
            text = res.read().decode("utf8")
            yield JsonExecutorResult(status=status, trace=trace, text=text)
        elif PlainTextExecutorResult.is_eligible(ctype):
            yield PlainTextExecutorResult(
                status=status, trace=trace, reader=res
            )
        else:
            raise unsupported_content_type_error(
                text=res.read().decode("utf8"),
                content_type=ctype,
                trace=trace,
            )
