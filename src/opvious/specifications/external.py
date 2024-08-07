from __future__ import annotations

import dataclasses
from typing import Optional

from ..executors import Executor, PlainTextExecutorResult


@dataclasses.dataclass(frozen=True)
class RemoteSpecification:
    """A model specification from a remote URL"""

    url: str
    """The specification's http(s) URL"""

    async def fetch_sources(self, executor: Executor) -> list[str]:
        async with executor.execute(
            result_type=PlainTextExecutorResult,
            url=self.url,
        ) as res:
            source = await res.text(assert_status=200)
        return [source]


@dataclasses.dataclass(frozen=True)
class FormulationSpecification:
    """A specification from an `Optimization Hub`_ formulation

    This type of specification allows queueing solves and is recommended for
    production use as it provides history and reproducibility when combined
    with tag names.

    .. _Optimization Hub: https://hub.cloud.opvious.io
    """

    formulation_name: str
    """The corresponding formulation's name"""

    tag_name: Optional[str] = None
    """The matching tag's name

    If absent, the formulation's default will be used.
    """
