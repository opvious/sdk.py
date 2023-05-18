from __future__ import annotations

import collections
import dataclasses
import glob
import logging
import os
from typing import Any, ClassVar, Iterable, Optional, Sequence

from ..common import Json, Setting, default_api_url, json_dict
from ..executors import (
    Executor,
    JsonExecutorResult,
    default_executor,
)


_logger = logging.getLogger(__name__)


_DEFAULT_TITLE = "<inline>"


@dataclasses.dataclass(frozen=True)
class LocalSpecificationSource:
    text: str
    title: str = _DEFAULT_TITLE


@dataclasses.dataclass(frozen=True)
class LocalSpecificationAnnotation:
    issue_count: int
    issues: Sequence[Sequence[LocalSpecificationIssue]]
    counts: Sequence[collections.Counter]


@dataclasses.dataclass(frozen=True)
class LocalSpecificationIssue:
    start_offset: int
    end_offset: int
    message: str
    code: str


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

    annotation: Optional[LocalSpecificationAnnotation] = None

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

    async def annotated(
        self,
        ignore_codes: Optional[Iterable[str]] = None,
        executor: Optional[Executor] = None,
    ) -> LocalSpecification:
        sources = self.sources
        codes = set(ignore_codes or [])

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
                if e["code"] in codes:
                    continue
                issues[e["index"]].append(_issue_from_json(e))
            annotation = LocalSpecificationAnnotation(
                issue_count=len(data["errors"]),
                issues=issues,
                counts=counts,
            )
        return dataclasses.replace(self, annotation=annotation)

    def _repr_markdown_(self) -> str:
        annotation = self.annotation
        sources = self.sources
        if annotation:
            for i, issues in enumerate(annotation.issues):
                if not issues:
                    continue
                messages = [f"  * [{i.code}] {i.message}" for i in issues]
                _logger.error(
                    "%s issue(s) in %s specification:\n%s",
                    len(issues),
                    sources[i].title,
                    "\n".join(messages),
                )
        return "\n\n---\n\n".join(
            _source_details(
                s,
                annotation.counts[i] if annotation else None,
                annotation.issues[i] if annotation else [],
                start_open=i == 0,
            )
            for i, s in enumerate(self.sources)
        )


def _source_details(
    source: LocalSpecificationSource,
    counts: Optional[collections.Counter],
    issues: Sequence[LocalSpecificationIssue],
    start_open=False,
) -> str:
    lines = ["<details open>" if start_open else "<details>"]
    if source.title:
        lines.append('<summary style="cursor:pointer;">&#9656; ')
        summary = source.title
        if counts:
            parens = ", ".join(f"{k}: {v}" for k, v in counts.most_common())
            summary += f" ({parens})"
        lines.append(summary)
        lines.append("</summary>\n")
    lines.append(_colorize(source.text, issues))
    lines.append("</details>")
    return "\n".join(lines)


def _issue_from_json(data: Json) -> LocalSpecificationIssue:
    rg = data["range"]
    return LocalSpecificationIssue(
        message=data["message"],
        start_offset=rg["start"]["offset"],
        end_offset=rg["end"]["offset"],
        code=data["code"],
    )


def _colorize(text: str, issues: Sequence[LocalSpecificationIssue]) -> str:
    if not issues:
        return text

    by_start = sorted(issues, key=lambda i: i.start_offset)
    cutoffs: list[int] = []
    end = 0
    for issue in by_start:
        if issue.code == "ERR_INVALID_SYNTAX":
            # These issues can't be easily colorized as they span name
            # boundaries
            continue
        if issue.start_offset <= end:
            end = max(issue.end_offset + 1, end)
        else:
            cutoffs.append(end)
            cutoffs.append(issue.start_offset)
            end = issue.end_offset + 1
    cutoffs.append(end)
    cutoffs.append(len(text))

    ret: list[str] = []
    for i, piece in enumerate(text[m:n] for m, n in zip(cutoffs, cutoffs[1:])):
        is_error = i % 2 == 1
        if is_error:
            ret.append(f"\\color{{orange}}{{{piece}}}")
        else:
            ret.append(piece)
    return "".join(ret)
