from typing import Union

from .local import (
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
)
from .external import FormulationSpecification, RemoteSpecification
from .notebook import load_notebook_specification


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
    "RemoteSpecification",
    "Specification",
    "load_notebook_specification",
]
