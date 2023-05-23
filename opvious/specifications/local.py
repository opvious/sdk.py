from __future__ import annotations

import collections
import dataclasses
import glob
import logging
import os
from typing import Iterable, Mapping, Optional, Sequence

from ..common import Json


_DEFAULT_TITLE = "untitled"


_logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LocalSpecificationSource:
    text: str
    title: str = _DEFAULT_TITLE


@dataclasses.dataclass(frozen=True)
class LocalSpecificationAnnotation:
    issue_count: int
    issues: Mapping[int, Sequence[LocalSpecificationIssue]]


@dataclasses.dataclass(frozen=True)
class LocalSpecificationIssue:
    source_index: int
    start_offset: int
    end_offset: int
    message: str
    code: str


def local_specification_issue_from_json(data: Json) -> LocalSpecificationIssue:
    rg = data["range"]
    return LocalSpecificationIssue(
        message=data["message"],
        source_index=data["index"],
        start_offset=rg["start"]["offset"],
        end_offset=rg["end"]["offset"],
        code=data["code"],
    )


@dataclasses.dataclass(frozen=True)
class LocalSpecification:
    """
    A local specification

    This type of specification cannot be used to start attempts.
    """

    sources: Sequence[LocalSpecificationSource]
    """The model's mathematical source definitions"""

    description: Optional[str] = None
    """Optional description"""

    annotation: Optional[LocalSpecificationAnnotation] = None
    """API-issued annotation"""

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

    def annotated(
        self, issues: Iterable[LocalSpecificationIssue]
    ) -> LocalSpecification:
        count = 0
        grouped: dict[
            int, list[LocalSpecificationIssue]
        ] = collections.defaultdict(list)
        for issue in issues:
            count += 1
            grouped[issue.source_index].append(issue)
        annotation = LocalSpecificationAnnotation(
            issues=dict(grouped),
            issue_count=count,
        )
        return dataclasses.replace(self, annotation=annotation)

    def _repr_markdown_(self) -> str:
        if self.annotation:
            issues = self.annotation.issues
            for index, group in issues.items():
                messages = [f"  [{i.code}] {i.message}" for i in group]
                _logger.error(
                    "%s issue(s) in specification '%s':\n%s",
                    len(group),
                    self.sources[index].title,
                    "\n".join(messages),
                )
        else:
            issues = {}
        return "\n\n---\n\n".join(
            _source_details(s, issues.get(i) or [])
            for i, s in enumerate(self.sources)
        )


_SUMMARY_STYLE = "".join(
    [
        "cursor: pointer;",
        "text-decoration: underline;",
        "text-decoration-style: dotted;",
    ]
)


def _source_details(
    source: LocalSpecificationSource,
    issues: Sequence[LocalSpecificationIssue],
    start_open=False,
) -> str:
    summary = source.title
    if issues:
        summary += " &#9888;"
    return "\n".join(
        [
            "<details open>" if start_open else "<details>",
            f'<summary style="{_SUMMARY_STYLE}">{summary}</summary>',
            _colorize(source.text, issues),
            "</details>",
        ]
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
