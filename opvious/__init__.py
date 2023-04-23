"""Opvious Python SDK"""

import logging

from .client import Client, ClientSetting
from .common import __version__
from .data.attempts import Attempt, AttemptNotification, AttemptRequest
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
    ConstraintRelaxation,
    RelaxationPenalty,
    Relaxation,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    SolveSummary,
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


__all__ = [
    "Attempt",
    "AttemptNotification",
    "AttemptRequest",
    "CancelledOutcome",
    "Client",
    "ClientSetting",
    "ConstraintRelaxation",
    "DimensionArgument",
    "Executor",
    "ExecutorError",
    "FailedOutcome",
    "FeasibleOutcome",
    "FormulationSpecification",
    "InfeasibleOutcome",
    "InlineSpecification",
    "Key",
    "KeyItem",
    "Label",
    "LocalSpecification",
    "Outcome",
    "Outline",
    "Relaxation",
    "RelaxationPenalty",
    "RemoteSpecification",
    "SolveInputs",
    "SolveOptions",
    "SolveOutputs",
    "SolveResponse",
    "SolveSummary",
    "SolveStatus",
    "SparseTensorArgument",
    "Specification",
    "Tensor",
    "TensorArgument",
    "UnboundedOutcome",
    "UnexpectedOutcomeError",
    "Value",
    "__version__",
    "aiohttp_executor",
    "default_executor",
    "outcome_status",
    "pyodide_executor",
    "urllib_executor",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
