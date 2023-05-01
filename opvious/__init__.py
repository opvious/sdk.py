"""Opvious Python SDK"""

import logging

from .client import Client, ClientSetting
from .common import __version__
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
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    SolveSummary,
    SolveStrategy,
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
    InlineSpecification,
    LocalSpecification,
    RemoteSpecification,
    Specification,
)
from .transform import (
    RelaxationPenalty,
    Transformation,
    ConstrainObjective,
    DensifyVariables,
    OmitConstraints,
    OmitObjectives,
    PinVariables,
    RelaxConstraints,
)


__all__ = [
    # Client
    "Client",
    "ClientSetting",
    # Executors
    "Executor",
    "ExecutorError",
    "aiohttp_executor",
    "default_executor",
    "pyodide_executor",
    "urllib_executor",
    # Specifications
    "FormulationSpecification",
    "InlineSpecification",
    "LocalSpecification",
    "RemoteSpecification",
    "Specification",
    # Solves and attempts
    "Attempt",
    "AttemptNotification",
    "DimensionArgument",
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
    "RelaxationPenalty",
    "Transformation",
    "ConstrainObjective",
    "DensifyVariables",
    "OmitConstraints",
    "OmitObjectives",
    "PinVariables",
    "RelaxConstraints",
    # Miscellaneous
    "__version__",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
