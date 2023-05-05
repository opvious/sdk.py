from __future__ import annotations

import dataclasses
import glob
import os
from typing import Optional, Union

from .executors import Executor, PlainTextExecutorResult


class AnonymousSpecification:
    """
    An unnamed model specification. Its sources will be fetched at runtime.
    This type of specification cannot be used to start attempts.
    """

    async def fetch_sources(self, executor: Executor) -> list[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class LocalSpecification(AnonymousSpecification):
    """A model specification from local files

    This is most useful during development, for example when running tests. For
    production use-cases prefer using a :class:`.FormulationSpecification` as
    it has built-in version control and supports starting attempts.
    """

    paths: list[str]
    """The list of of paths which define the specification

    The specification may be split into multiple files. These may appear here
    in any order, the definitions will be sorted automatically.
    """

    @classmethod
    def globs(
        cls, *likes: str, root: Optional[str] = None
    ) -> LocalSpecification:
        """Creates a local specification from file globs

        As a convenience the root can be a file's name, in which case it will
        be interpreted as its parent directory (this is typically handy when
        used with `__file__`).
        """
        if root:
            root = os.path.realpath(root)
            if os.path.isfile(root):
                root = os.path.dirname(root)
        paths = []
        for like in likes:
            for path in glob.iglob(like, root_dir=root, recursive=True):
                if root:
                    path = os.path.join(root, path)
                paths.append(path)
        return LocalSpecification(paths)

    async def fetch_sources(self, _executor: Executor) -> list[str]:
        sources = []
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as reader:
                sources.append(reader.read())
        return sources


@dataclasses.dataclass(frozen=True)
class InlineSpecification(AnonymousSpecification):
    """A model specification from inline sources

    This can be useful for writing unit-tests or defining toy models. In
    general prefer :class:`.LocalSpecification` as it allows validating the
    specification via the `CLI <https://www.npmjs.com/package/opvious-cli>`_.
    """

    sources: list[str]
    """The list of sources"""

    async def fetch_sources(self, _executor: Executor) -> list[str]:
        return self.sources


_EXAMPLE_URL_PREFIX = (
    "https://raw.githubusercontent.com/opvious/examples/main/sources"  # noqa
)


@dataclasses.dataclass(frozen=True)
class RemoteSpecification(AnonymousSpecification):
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


Specification = Union[
    AnonymousSpecification,
    FormulationSpecification,
]
