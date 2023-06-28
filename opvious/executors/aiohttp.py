import aiohttp
import brotli  # type: ignore
import contextlib
import logging
from typing import AsyncIterator, Optional

from .common import (
    CONTENT_TYPE_HEADER,
    Executor,
    ExecutorResult,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
    Headers,
    TRACE_HEADER,
    unexpected_response_error,
    unsupported_content_type_error,
)


_logger = logging.getLogger(__name__)


_BROTLI_QUALITY = 4


_COMPRESSION_THRESHOLD = 2**16


_READ_BUFFER_SIZE = 2**26  # 512MiB


_REQUEST_TIMEOUT_SECONDS = 900  # 15 minutes


class AiohttpExecutor(Executor):
    """`aiohttp`-powered executor"""

    def __init__(self, endpoint: str, authorization: Optional[str] = None):
        super().__init__(
            variant="aiohttp",
            endpoint=endpoint,
            authorization=authorization,
            supports_streaming=True,
        )

    @contextlib.asynccontextmanager
    async def _send(
        self, url: str, method: str, headers: Headers, body: Optional[bytes]
    ) -> AsyncIterator[ExecutorResult]:
        if body and len(body) > _COMPRESSION_THRESHOLD:
            headers["content-encoding"] = "br"
            compressed_body = brotli.compress(
                body,
                mode=brotli.MODE_TEXT,
                quality=_BROTLI_QUALITY,
            )
            _logger.debug(
                "Compressed request body. [size=%s]", len(compressed_body)
            )
            body = compressed_body
        try:
            async with aiohttp.ClientSession(
                headers=headers,
                read_bufsize=_READ_BUFFER_SIZE,
                timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS),
            ) as session:
                async with session.request(
                    url=url, method=method, data=body
                ) as res:
                    status = res.status
                    trace = res.headers.get(TRACE_HEADER)
                    ctype = res.headers.get(CONTENT_TYPE_HEADER)
                    if JsonExecutorResult.is_eligible(ctype):
                        text = await res.text()
                        yield JsonExecutorResult(
                            status=status,
                            trace=trace,
                            text=text,
                        )
                    elif JsonSeqExecutorResult.is_eligible(ctype):
                        yield JsonSeqExecutorResult(
                            status=status,
                            trace=trace,
                            reader=res.content,
                        )
                    elif PlainTextExecutorResult.is_eligible(ctype):
                        yield PlainTextExecutorResult(
                            status=status,
                            trace=trace,
                            reader=res.content,
                        )
                    else:
                        raise unsupported_content_type_error(
                            content_type=ctype,
                            trace=trace,
                        )
        except aiohttp.ClientResponseError as err:
            trace = None
            if isinstance(err.headers, list):
                trace = next(
                    (v for (k, v) in err.headers if k == TRACE_HEADER), None
                )
            raise unexpected_response_error(message=err.message, trace=trace)
