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
class InlineSpecification(AnonymousSpecification):
    """
    A model specification from inline sources. See also `LocalSpecification`
    for reading specifications from locally stored files.
    """

    sources: list[str]

    async def fetch_sources(self, _executor: Executor) -> list[str]:
        return self.sources


@dataclasses.dataclass(frozen=True)
class LocalSpecification(AnonymousSpecification):
    """
    A model specification from local files. See also `InlineSpecification` for
    creating specifications directly from strings.
    """

    paths: list[str]

    @classmethod
    def globs(
        cls, *likes: str, root: Optional[str] = None
    ) -> LocalSpecification:
        """
        Creates a local specification from file globs. As a convenience the
        root can be a file's name, in which case it will be interpreted as its
        parent directory (this is typically handy when used with `__file__`).
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


_EXAMPLE_URL_PREFIX = (
    "https://raw.githubusercontent.com/opvious/examples/main/sources"  # noqa
)


@dataclasses.dataclass(frozen=True)
class RemoteSpecification(AnonymousSpecification):
    """A model specification from a remote URL"""

    url: str

    @classmethod
    def example(cls, name: str):
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
    """
    A specification from a stored formulation. This type of specification
    allows starting attempts and is recommended for production use as it
    provides history and reproducibility when combined with tag names.
    """

    formulation_name: str
    tag_name: Optional[str] = None


Specification = Union[
    AnonymousSpecification,
    FormulationSpecification,
]
