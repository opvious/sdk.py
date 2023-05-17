from __future__ import annotations

import dataclasses
from typing import Optional

from ..executors import Executor, PlainTextExecutorResult


_EXAMPLE_URL_PREFIX = (
    "https://raw.githubusercontent.com/opvious/examples/main/sources"  # noqa
)


@dataclasses.dataclass(frozen=True)
class RemoteSpecification:
    """A model specification from a remote URL

    This is typically useful for examples.
    """

    url: str
    """The specification's http(s) URL"""

    @classmethod
    def example(cls, name: str):
        """Returns a standard example's specification

        Standard examples are available here:
        https://github.com/opvious/examples/tree/main/sources
        """
        return RemoteSpecification(url=f"{_EXAMPLE_URL_PREFIX}/{name}.md")

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

    This type of specification allows starting attempts and is recommended for
    production use as it provides history and reproducibility when combined
    with tag names.

    `This GitHub action
    <https://github.com/opvious/register-specification-action>`_ provides a
    convenient way to automatically create formulations from CI workflows.

    .. _Optimization Hub: https://hub.beta.opvious.io
    """

    formulation_name: str
    """The corresponding formulation's name"""

    tag_name: Optional[str] = None
    """The matching tag's name

    If absent, the formulation's default will be used.
    """
