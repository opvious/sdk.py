import logging

from .client import Client, ClientSetting, Problem
from .common import __version__
from .data.attempts import Attempt, AttemptNotification
from .data.outcomes import (
    AbortedOutcome,
    CancelledOutcome,
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
from .data.solves import (
    EpsilonConstraint,
    Solution,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveSummary,
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
    # Solves and attempts
    "Attempt",
    "AttemptNotification",
    "DimensionArgument",
    "EpsilonConstraint",
    "Key",
    "KeyItem",
    "Label",
    "Outline",
    "Problem",
    "Solution",
    "SolveInputs",
    "SolveOptions",
    "SolveOutputs",
    "SolveStatus",
    "SolveStrategy",
    "SolveSummary",
    "SparseTensorArgument",
    "Target",
    "Tensor",
    "TensorArgument",
    "Value",
    # Outcomes
    "AbortedOutcome",
    "CancelledOutcome",
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
