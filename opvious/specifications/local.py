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

    def _repr_markdown_(self) -> str:
        return _source_details(self, [], True)


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


_SPECIFICATION_STYLE = " ".join(
    [
        "margin-top: 1em;",
        "margin-bottom: 1em;",
    ]
)


_SUMMARY_STYLE = " ".join(
    [
        "cursor: pointer;",
        "text-decoration: underline;",
        "text-decoration-style: dotted;",
    ]
)


@dataclasses.dataclass(frozen=True)
class LocalSpecification:
    """
    A local specification

    Instances are integrated with IPython's `rich display capabilities`_ and
    will automatically render their LaTeX sources when output in notebooks.

    This type of specification cannot be used to start attempts.

    .. _rich display capabilities: https://ipython.readthedocs.io/en/stable/config/integrating.html#rich-display  # noqa
    """

    sources: Sequence[LocalSpecificationSource]
    """The model's mathematical source definitions"""

    description: Optional[str] = None
    """Optional description"""

    annotation: Optional[LocalSpecificationAnnotation] = None
    """API-issued annotation

    This field is typically generated automatically by clients'
    :meth:`~opvious.Client.annotate_specification` method. When any issues are
    detected, the specification's pretty-printed representation will highlight
    any errors.
    """

    def source(self, title: Optional[str] = None) -> LocalSpecificationSource:
        """Returns the first source, optionally matching a given title"""
        if title is None:
            return self.sources[0]
        for source in self.sources:
            if source.title == title:
                return source
        raise Exception(f"Missing source for title {title}")

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
                messages = [f"\t* {i.message} [{i.code}]" for i in group]
                _logger.error(
                    "%s issue(s) in specification '%s':\n%s",
                    len(group),
                    self.sources[index].title,
                    "\n".join(messages),
                )
        else:
            issues = {}
        contents = "\n---\n".join(
            _source_details(s, issues.get(i) or [], i == 0)
            for i, s in enumerate(self.sources)
        )
        return f'<div style="{_SPECIFICATION_STYLE}">{contents}</div>'


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
            "",
            "<details open>" if start_open else "<details>",
            f'<summary style="{_SUMMARY_STYLE}">{summary}</summary>',
            '<div style="margin-top: 1em;">',
            _colorize(source.text, issues),
            "</div>",
            "</details>",
            "",
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
