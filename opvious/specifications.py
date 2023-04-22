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
from typing import Optional, Union

from .executors import Executor, PlainTextExecutorResult


class AnonymousSpecification:
    async def fetch_sources(self, executor: Executor) -> list[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class InlineSpecification(AnonymousSpecification):
    """A model specification from inline sources"""

    sources: list[str]

    async def fetch_sources(self, _executor: Executor) -> list[str]:
        return self.sources


_EXAMPLE_URL_PREFIX = (
    "https://raw.githubusercontent.com/opvious/examples/main/sources"  # noqa
)


@dataclasses.dataclass(frozen=True)
class RemoteSpecification(AnonymousSpecification):
    """A model specification from a URL"""

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
    """A model specification from inline sources"""

    formulation_name: str
    tag_name: Optional[str] = None


Specification = Union[
    AnonymousSpecification,
    FormulationSpecification,
]