from __future__ import annotations

import collections
import dataclasses
import glob
import logging
import os
from typing import Any, ClassVar, Optional, Sequence

from ..common import Json, Setting, default_api_url, json_dict
from ..executors import (
    Executor,
    JsonExecutorResult,
    default_executor,
    run_sync,
)


_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LocalSpecificationSource:
    text: str
    title: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class LocalSpecificationAnnotation:
    counts: Sequence[collections.Counter]
    issues: Sequence[Sequence[LocalSpecificationIssue]]


@dataclasses.dataclass(frozen=True)
class LocalSpecificationIssue:
    start_offset: int
    end_offset: int
    message: str


@dataclasses.dataclass(frozen=True)
class LocalSpecification:
    """
    A local specification

    This type of specification cannot be used to start attempts.
    """

    __executor: ClassVar[Executor] = default_executor(
        default_api_url(Setting.DOMAIN.read())
    )

    sources: Sequence[LocalSpecificationSource]

    @classmethod
    def set_default_executor(cls, executor: Executor) -> None:
        cls.__executor = executor

    @classmethod
    def inline(cls, *texts: str) -> LocalSpecification:
        """Creates a local specification from strings"""
        sources = [LocalSpecificationSource(s) for s in texts]
        return LocalSpecification(sources)

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
        sources: list[LocalSpecificationSource] = []
        for like in likes:
            for path in glob.iglob(like, root_dir=root, recursive=True):
                if root:
                    path = os.path.join(root, path)
                with open(path, "r", encoding="utf-8") as reader:
                    sources.append(
                        LocalSpecificationSource(
                            text=reader.read(),
                            title=path,
                        )
                    )
        return LocalSpecification(sources)

    async def fetch_annotation(
        self, executor: Optional[Executor] = None
    ) -> LocalSpecificationAnnotation:
        sources = self.sources
        if not executor:
            executor = self.__executor
        async with executor.execute(
            result_type=JsonExecutorResult,
            url="/sources/parse",
            method="POST",
            json_data=json_dict(sources=[s.text for s in sources]),
        ) as res:
            data = res.json_data()
            counts: Sequence[Any] = [collections.Counter() for _ in sources]
            for s in data["slices"]:
                category = s["definition"]["category"].lower()
                counts[s["index"]][category] += 1
            issues: Sequence[Any] = [[] for _ in sources]
            for e in data["errors"]:
                issues[e["index"]].append(_issue_from_json(e))
            return LocalSpecificationAnnotation(counts, issues)

    def _repr_markdown_(self) -> str:
        try:
            annotation = run_sync(self.fetch_annotation, self.sources)
        except Exception:
            _logger.warn("Unable to annotate specification", exc_info=True)
            annotation = None
        else:
            for issue in annotation.issues:
                _logger.error("Invalid specification: %s", issue.message)
        return "\n---\n".join(
            _source_details(
                s,
                annotation.counts[i] if annotation else None,
                annotation.issues[i] if annotation else [],
            )
            for i, s in enumerate(self.sources)
        )


def _source_details(
    source: LocalSpecificationSource,
    counts: Optional[collections.Counter],
    issues: Sequence[LocalSpecificationIssue],
) -> str:
    lines = ["<details>"]
    if source.title:
        lines.append("<summary><em>")
        if source.title:
            lines.append(source.title)
        else:
            lines.append("<inline>")
        if counts:
            parens = ", ".join(f"{k}: {v}" for k, v in counts.most_common())
            lines.append(f" ({parens})")
        lines.append("<\\em><\\summary>")
    lines.append(source.text)  # TODO: Annotate with issues
    lines.append("<\\details>")
    return "\n".join(lines)


def _issue_from_json(data: Json) -> LocalSpecificationIssue:
    rg = data["range"]
    return LocalSpecificationIssue(
        message=data["message"],
        start_offset=rg["start"]["offset"],
        end_offset=rg["end"]["offset"],
    )
