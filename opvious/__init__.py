import logging

from .client import Client
from .common import __version__, Setting
from .data.attempts import Attempt, AttemptNotification
from .data.outcomes import (
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
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
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
from .executors import (
    Executor,
    ExecutorError,
    aiohttp_executor,
    default_executor,
    pyodide_executor,
    urllib_executor,
)
from .specifications import (
    FormulationSpecification,
    LocalSpecification,
    LocalSpecificationIssue,
    LocalSpecificationSource,
    RemoteSpecification,
    Specification,
    load_notebook_models,
    model as modeling,
)
from .transformations import (
    ConstrainObjective,
    DensifyVariables,
    OmitConstraints,
    OmitObjectives,
    PinVariables,
    RelaxConstraints,
    RelaxationPenalty,
    Transformation,
)


__all__ = [
    # Configuration
    "Setting",
    # Client
    "Client",
    # Executors
    "Executor",
    "ExecutorError",
    "aiohttp_executor",
    "default_executor",
    "pyodide_executor",
    "urllib_executor",
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
    "SolveInputs",
    "SolveOptions",
    "SolveOutputs",
    "SolveResponse",
    "SolveStatus",
    "SolveStrategy",
    "SolveSummary",
    "SparseTensorArgument",
    "Target",
    "Tensor",
    "TensorArgument",
    "Value",
    # Outcomes
    "CancelledOutcome",
    "FailedOutcome",
    "FeasibleOutcome",
    "InfeasibleOutcome",
    "Outcome",
    "UnboundedOutcome",
    "UnexpectedOutcomeError",
    "outcome_status",
    # Transformations
    "ConstrainObjective",
    "DensifyVariables",
    "OmitConstraints",
    "OmitObjectives",
    "PinVariables",
    "RelaxConstraints",
    "RelaxationPenalty",
    "Transformation",
    # Miscellaneous
    "__version__",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
