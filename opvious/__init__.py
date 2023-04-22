"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
"""

import logging

from .client import Client, ClientSettings
from .common import __version__
from .data.attempts import Attempt, AttemptRequest, Notification
from .data.outcomes import (
    CancelledOutcome,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    Outcome,
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
    "AttemptRequest",
    "CancelledOutcome",
    "Client",
    "ClientSettings",
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
    "Notification",
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
    "pyodide_executor",
    "urllib_executor",
]


logging.getLogger(__name__).addHandler(logging.NullHandler())
