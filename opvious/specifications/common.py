from __future__ import annotations

import dataclasses
from typing import Sequence

from ..common import Json


@dataclasses.dataclass(init=False)
class SpecificationValidationError(Exception):
    issues: Sequence[SpecificationSourceIssue]

    def __init__(self, issues: Sequence[SpecificationSourceIssue]) -> None:
        super().__init__(f"Invalid specification: {len(issues)} issue(s)")
        self.issues = issues
        print(issues)


@dataclasses.dataclass(frozen=True)
class SpecificationSourceIssue:
    index: int
    start_offset: int
    end_offset: int
    message: str


def source_issue_from_json(data: Json) -> SpecificationSourceIssue:
    rg = data["range"]
    return SpecificationSourceIssue(
        message=data["message"],
        index=data["index"],
        start_offset=rg["start"]["offset"],
        end_offset=rg["end"]["offset"],
    )
