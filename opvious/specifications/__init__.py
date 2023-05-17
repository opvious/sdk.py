from typing import Union

from .local import (
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
)
from .external import FormulationSpecification, RemoteSpecification


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
]
