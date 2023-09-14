import logging

from .client import Client, ClientSetting, Problem
from .common import __version__
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
    Solution,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    ProblemSummary,
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
from . import executors
from . import modeling
from .specifications import (
    FormulationSpecification,
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
    RemoteSpecification,
    Specification,
    load_notebook_models,
)
from . import transformations


__all__ = [
    # Client
    "Client",
    "ClientSetting",
    # Executors
    "executors",
    # Specifications
    "FormulationSpecification",
    "LocalSpecification",
    "LocalSpecificationIssue",
    "LocalSpecificationSource",
    "RemoteSpecification",
    "Specification",
    "load_notebook_models",
    "modeling",
    # Solves
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
