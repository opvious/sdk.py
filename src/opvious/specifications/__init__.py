from typing import Union

from .local import (
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
    LocalSpecificationStyle,
    local_specification_issue_from_json,
)
from .external import FormulationSpecification, RemoteSpecification
from .notebook import load_notebook_models


Specification = Union[
    LocalSpecification,
    RemoteSpecification,
    FormulationSpecification,
]


__all__ = [
    "FormulationSpecification",
    "LocalSpecification",
    "LocalSpecificationIssue",
    "LocalSpecificationSource",
    "LocalSpecificationStyle",
    "RemoteSpecification",
    "Specification",
    "local_specification_issue_from_json",
    "load_notebook_models",
]
