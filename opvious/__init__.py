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

from .client import Client, Settings
from .common import __version__
from .executors import (
    ApiError,
    Executor,
    aiohttp_executor,
    default_executor,
    pyodide_executor,
    urllib_executor,
)
from .data import (
    Attempt,
    AttemptRequest,
    CancelledOutcome,
    ConstraintRelaxation,
    DimensionArgument,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    Key,
    KeyItem,
    Label,
    Notification,
    Outcome,
    Penalty,
    Relaxation,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    SparseTensorArgument,
    Summary,
    Tensor,
    TensorArgument,
    UnboundedOutcome,
    UnexpectedOutcomeError,
    Value,
)


__all__ = [
    "ApiError",
    "Attempt",
    "AttemptRequest",
    "CancelledOutcome",
    "Client",
    "ConstraintRelaxation",
    "DimensionArgument",
    "Executor",
    "FailedOutcome",
    "FeasibleOutcome",
    "InfeasibleOutcome",
    "Key",
    "KeyItem",
    "Label",
    "Notification",
    "Outcome",
    "Penalty",
    "Relaxation",
    "Settings",
    "SolveInputs",
    "SolveOptions",
    "SolveOutputs",
    "SolveResponse",
    "SparseTensorArgument",
    "Summary",
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
