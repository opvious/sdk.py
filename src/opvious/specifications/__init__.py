from .external import FormulationSpecification, RemoteSpecification
from .local import (
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
    LocalSpecificationStyle,
    local_specification_issue_from_json,
)


type Specification = (
    LocalSpecification | RemoteSpecification | FormulationSpecification
)


__all__ = [
    "FormulationSpecification",
    "LocalSpecification",
    "LocalSpecificationIssue",
    "LocalSpecificationSource",
    "LocalSpecificationStyle",
    "RemoteSpecification",
    "Specification",
    "load_notebook_models",
    "local_specification_issue_from_json",
]
