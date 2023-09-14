import logging

from .client import Client, ClientSetting, Problem
from .common import __version__
from .data.outcomes import (
    AbortedOutcome,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    Outcome,
    outcome_status,
    SolveStatus,
    UnboundedOutcome,
    UnexpectedOutcomeError,
)
from .data.outlines import Label, Outline
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
    "Outline",
    "Problem",
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
    "Outcome",
    "UnboundedOutcome",
    "UnexpectedOutcomeError",
    "outcome_status",
    # Transformations
    "transformations",
    # Miscellaneous
    "__version__",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
