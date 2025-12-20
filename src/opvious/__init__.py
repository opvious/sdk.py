import logging

from . import executors, modeling, transformations
from .client import DEMO_ENDPOINT, Client, Problem
from .common import Annotation, __version__
from .data.outcomes import (
    AbortedOutcome,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    SolveOutcome,
    SolveStatus,
    UnboundedOutcome,
    UnexpectedSolveOutcomeError,
    solve_outcome_status,
)
from .data.outlines import Label, ProblemOutline
from .data.queued_solves import QueuedSolve, SolveNotification
from .data.solves import (
    EpsilonConstraint,
    ProblemSummary,
    Solution,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveStrategy,
    Target,
)
from .data.tensors import (
    DimensionArgument,
    Key,
    KeyItem,
    SparseTensorArgument,
    Tensor,
    TensorArgument,
    Value,
)
from .notebooks import load_notebook_models
from .specifications import (
    FormulationSpecification,
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
    LocalSpecificationStyle,
    RemoteSpecification,
    Specification,
)


__all__ = [
    # Client
    "Client",
    "DEMO_ENDPOINT",
    # Executors
    "executors",
    # Specifications
    "FormulationSpecification",
    "LocalSpecification",
    "LocalSpecificationIssue",
    "LocalSpecificationSource",
    "LocalSpecificationStyle",
    "RemoteSpecification",
    "Specification",
    "load_notebook_models",
    "modeling",
    # Solves
    "Annotation",
    "DimensionArgument",
    "EpsilonConstraint",
    "Key",
    "KeyItem",
    "Label",
    "Problem",
    "ProblemOutline",
    "ProblemSummary",
    "QueuedSolve",
    "Solution",
    "SolveInputs",
    "SolveNotification",
    "SolveOptions",
    "SolveOutputs",
    "SolveStatus",
    "SolveStrategy",
    "SparseTensorArgument",
    "Target",
    "Tensor",
    "TensorArgument",
    "Value",
    # Outcomes
    "AbortedOutcome",
    "FailedOutcome",
    "FeasibleOutcome",
    "InfeasibleOutcome",
    "SolveOutcome",
    "UnboundedOutcome",
    "UnexpectedSolveOutcomeError",
    "solve_outcome_status",
    # Transformations
    "transformations",
    # Miscellaneous
    "__version__",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
