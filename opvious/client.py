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

import backoff
from datetime import datetime, timezone
import enum
import json
import humanize
import logging
import os
from typing import (
    Any,
    cast,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from .common import format_percent, strip_nones
from .data import (
    Attempt,
    AttemptRequest,
    CancelledOutcome,
    DimensionArgument,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    Label,
    Notification,
    Outcome,
    Outline,
    Relaxation,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    solve_options_to_json,
    Summary,
    Tensor,
    TensorArgument,
    UnboundedOutcome,
)
from .executors import (
    default_executor,
    Executor,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
)


_logger = logging.getLogger(__name__)


_DEFAULT_DOMAIN = "beta.opvious.io"


class Settings(enum.Enum):
    """Environment variable names"""

    TOKEN = "OPVIOUS_TOKEN"
    DOMAIN = "OPVIOUS_DOMAIN"


class Client:
    """Opvious API client"""

    def __init__(self, executor: Executor, api_url: str, hub_url: str):
        self._executor = executor
        self._api_url = api_url
        self._hub_url = hub_url

    def __repr__(self) -> str:
        fields = [
            f"executor_class={self._executor.__class__.__name__}",
            f"api_url={json.dumps(self._api_url)}",
        ]
        return f"<opvious.Client {' '.join(fields)}>"

    @classmethod
    def from_token(cls, token: str, domain: Optional[str] = None) -> "Client":
        """
        Creates a client from an API token. You can use an empty token to
        create an unauthenticated client with limited functionality.
        """
        token = token.strip()
        authorization = None
        if token:
            authorization = token if " " in token else f"Bearer {token}"
        api_url = f"https://api.{domain or _DEFAULT_DOMAIN}"
        return Client(
            executor=default_executor(
                api_url=api_url,
                authorization=authorization,
            ),
            api_url=api_url,
            hub_url=f"https://hub.{domain}",
        )

    @classmethod
    def from_environment(
        cls, env=os.environ, require_authenticated=False
    ) -> "Client":
        """
        Creates a client from environment variables. If present, the token
        should contain a valid API token. A custom domain can optionally be
        set.
        """
        token = env.get(Settings.TOKEN.value, "")
        if not token and require_authenticated:
            raise Exception(
                f"Missing or empty {Settings.TOKEN.value} environment variable"
            )
        return Client.from_token(
            token=token, domain=env.get(Settings.DOMAIN.value)
        )

    @property
    def authenticated(self):
        """
        Returns true if the client was created with a non-empty API token.
        """
        return self._executor.authenticated

    async def inspect(
        self,
        sources: Optional[list[str]] = None,
        formulation_name: Optional[str] = None,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
    ) -> str:
        """Inspects an optimization problem, returning its underlying solver
        instructions.
        """
        body, _outline = await self._assemble_solve_request(
            sources=sources,
            formulation_name=formulation_name,
            tag_name=tag_name,
            parameters=parameters,
            dimensions=dimensions,
        )
        async with self._executor.execute(
            result_type=PlainTextExecutorResult,
            path="/solves/inspect/instructions",
            method="POST",
            json_data={
                "runRequest": body,
            },
        ) as res:
            lines = []
            async for line in res.lines():
                if line.startswith("\\"):
                    _logger.debug(line[2:].strip())
                else:
                    lines.append(line)
            return "".join(lines)

    async def solve(
        self,
        sources: Optional[list[str]] = None,
        formulation_name: Optional[str] = None,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        relaxation: Optional[Relaxation] = None,
        options: Optional[SolveOptions] = None,
    ) -> SolveResponse:
        """Solves an optimization problem. See also `start_attempt` for an
        alternative for long-running solves.
        """
        body, outline = await self._assemble_solve_request(
            sources=sources,
            formulation_name=formulation_name,
            tag_name=tag_name,
            parameters=parameters,
            dimensions=dimensions,
            relaxation=relaxation,
            options=options,
        )
        solved_data = None
        async with self._executor.execute(
            result_type=JsonSeqExecutorResult,
            path="/solves/run",
            method="POST",
            json_data=body,
        ) as res:
            async for data in res.json_seq_data():
                kind = data["kind"]
                if kind == "reifying":
                    progress = data["progress"]
                    if progress["kind"] == "constraint":
                        summary = progress["summary"]
                        _logger.debug(
                            "Reified constraint %r. [columns=%s, rows=%s]",
                            summary["label"],
                            summary["columnCount"],
                            summary["rowCount"],
                        )
                elif kind == "reified":
                    summary = Summary.from_json(data["summary"])
                    density = summary.density()
                    _logger.info(
                        (
                            "Solving problem... [columns=%s, rows=%s, "
                            "weights=%s (%s)]"
                        ),
                        summary.column_count,
                        summary.row_count,
                        summary.weight_count,
                        format_percent(density),
                    )
                elif kind == "solving":
                    progress = data["progress"]
                    iter_count = progress.get("lpIterationCount")
                    gap = progress.get("relativeGap")
                    if iter_count is not None:
                        _logger.info(
                            "Solve in progress... [iters=%s, cuts=%s, gap=%s]",
                            iter_count,
                            progress.get("cutCount"),
                            "n/a" if gap is None else format_percent(gap),
                        )
                elif kind == "solved":
                    _logger.debug("Downloaded outputs.")
                    reached_at = datetime.now(timezone.utc)
                    solved_data = data
                elif kind == "error":
                    message = data["error"]["message"]
                    raise Exception(f"Solve failed: {message}")
                else:
                    raise Exception(f"Unexpected response: {json.dumps(data)}")
        if not solved_data:
            raise Exception("Solve terminated without any data")

        # Finally we gather the outputs
        outcome_data = solved_data["outcome"]
        status = outcome_data["status"]
        if status == "INFEASIBLE":
            outcome = cast(Outcome, InfeasibleOutcome(reached_at))
        elif status == "UNBOUNDED":
            outcome = UnboundedOutcome(reached_at)
        else:
            outcome = FeasibleOutcome(
                reached_at=reached_at,
                is_optimal=status == "OPTIMAL",
                objective_value=outcome_data.get("objectiveValue"),
                relative_gap=outcome_data.get("relativeGap"),
            )
        outputs = None
        if isinstance(outcome, FeasibleOutcome):
            outputs = SolveOutputs.from_json(
                data=solved_data["outputs"],
                outline=outline,
            )
        return SolveResponse(
            status=status,
            outcome=outcome,
            summary=summary,
            outputs=outputs,
        )

    async def _assemble_solve_request(
        self,
        sources: Optional[list[str]] = None,
        formulation_name: Optional[str] = None,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        options: Optional[SolveOptions] = None,
        relaxation: Union[None, Relaxation] = None,
    ) -> Tuple[Any, Outline]:
        # First we fetch the outline to validate/coerce inputs later on
        if formulation_name:
            if sources:
                raise Exception(
                    "Sources and formulation name are mutually exclusive"
                )
            outline, _tag = await self._fetch_formulation_outline(
                formulation_name, tag_name
            )
            formulation = strip_nones(
                {
                    "name": formulation_name,
                    "specificationTagName": tag_name,
                }
            )
        else:
            if not sources:
                raise Exception("Sources or formulation name must be set")
            outline = await self._fetch_sources_outline(sources)
            formulation = {"sources": sources}

        # Then we assemble the inputs
        builder = _SolveInputsBuilder(outline=outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        inputs = builder.build()
        _logger.info(
            "Validated inputs. [parameters=%s]",
            builder.parameter_entry_count,
        )

        # Finally we put everything together
        body = {
            "formulation": formulation,
            "inputs": strip_nones(
                {
                    "dimensions": inputs.raw_dimensions,
                    "parameters": inputs.raw_parameters,
                }
            ),
            "options": solve_options_to_json(options, relaxation),
        }
        return (body, outline)

    async def _fetch_sources_outline(self, sources: list[str]) -> Outline:
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            path="/sources/parse",
            method="POST",
            json_data={"sources": sources, "outline": True},
        ) as res:
            outline_data = res.json_data()
        errors = outline_data.get("errors")
        if errors:
            raise Exception(f"Invalid sources: {json.dumps(errors)}")
        return Outline.from_json(outline_data["outline"])

    async def _fetch_formulation_outline(
        self, name: str, tag_name: Optional[str] = None
    ) -> Tuple[Outline, str]:
        data = await self._executor.execute_graphql_query(
            query="@FetchOutline",
            variables={"formulationName": name, "tagName": tag_name},
        )
        formulation = data.get("formulation")
        if not formulation:
            raise Exception("No matching formulation found")
        tag = formulation.get("tag")
        if not tag:
            raise Exception("No matching specification found")
        spec = tag["specification"]
        outline = Outline.from_json(spec["outline"])
        return (outline, tag["name"])

    async def prepare_attempt_request(
        self,
        formulation_name: str,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
    ) -> AttemptRequest:
        """Assembles and validates inputs for a given formulation. The returned
        object can be used to start an asynchronous solve via `start_attempt`.
        """
        outline, tag = await self._fetch_formulation_outline(
            formulation_name, tag_name
        )
        builder = _SolveInputsBuilder(outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        return AttemptRequest(
            formulation_name=formulation_name,
            tag_name=tag,
            inputs=builder.build(),
        )

    async def start_attempt(
        self,
        request: AttemptRequest,
        options: Optional[SolveOptions] = None,
        relaxation: Union[None, Relaxation] = None,
        pinned_variables: Optional[Mapping[Label, TensorArgument]] = None,
    ) -> Attempt:
        """Starts a new asynchronous solve attempt."""
        pins = []
        if pinned_variables:
            for label, arg in pinned_variables.items():
                var_outline = request.inputs.outline.variables.get(label)
                if not var_outline:
                    raise Exception(f"Unknown variable {label}")
                tensor = Tensor.from_argument(
                    arg, len(var_outline.bindings), var_outline.is_indicator()
                )
                if tensor.default_value:
                    raise Exception("Pinned variables may not have defaults")
                pins.append({"label": label, "entries": tensor.entries})

        async with self._executor.execute(
            result_type=JsonExecutorResult,
            path="/attempts/start",
            method="POST",
            json_data={
                "formulationName": request.formulation_name,
                "specificationTagName": request.tag_name,
                "inputs": strip_nones(
                    {
                        "dimensions": request.inputs.raw_dimensions,
                        "parameters": request.inputs.raw_parameters,
                        "pinnedVariables": pins,
                    }
                ),
                "options": solve_options_to_json(options, relaxation),
            },
        ) as res:
            uuid = res.json_data()["uuid"]
        return Attempt(
            uuid=uuid,
            started_at=datetime.now(timezone.utc),
            outline=request.inputs.outline,
            url=self._attempt_url(uuid),
        )

    async def load_attempt(self, uuid: str) -> Optional[Attempt]:
        """Loads an existing attempt from its UUID."""
        data = await self._executor.execute_graphql_query(
            query="@FetchAttempt",
            variables={"uuid": uuid},
        )
        attempt = data["attempt"]
        if not attempt:
            return None
        return Attempt.from_graphql(
            data=attempt,
            outline=Outline.from_json(attempt["outline"]),
            url=self._attempt_url(uuid),
        )

    def _attempt_url(self, uuid) -> str:
        return self._hub_url + "/attempts/" + uuid

    async def cancel_attempt(self, uuid: str) -> bool:
        """Cancels a running attempt."""
        data = await self._executor.execute_graphql_query(
            query="@CancelAttempt",
            variables={"uuid": uuid},
        )
        return bool(data["cancelAttempt"])

    async def poll_attempt(
        self, attempt: Attempt
    ) -> Union[Notification, Outcome]:
        """Polls an attempt for its outcome or latest progress notification."""
        data = await self._executor.execute_graphql_query(
            query="@PollAttempt",
            variables={"uuid": attempt.uuid},
        )
        attempt_data = data["attempt"]
        status = attempt_data["status"]
        if status == "PENDING":
            edges = attempt_data["notifications"]["edges"]
            return Notification.from_graphql(
                dequeued=bool(attempt_data["dequeuedAt"]),
                data=edges[0]["node"] if edges else None,
            )
        reached_at = datetime.fromisoformat(attempt_data["endedAt"])
        if status == "CANCELLED":
            return cast(Outcome, CancelledOutcome(reached_at))
        if status == "INFEASIBLE":
            return InfeasibleOutcome(reached_at)
        if status == "UNBOUNDED":
            return UnboundedOutcome(reached_at)
        outcome = attempt_data["outcome"]
        if status == "ERRORED":
            return FailedOutcome.from_graphql(reached_at, outcome)
        if status == "FEASIBLE" or status == "OPTIMAL":
            return FeasibleOutcome.from_graphql(reached_at, outcome)
        raise Exception(f"Unexpected status {status}")

    @backoff.on_predicate(
        backoff.fibo,
        lambda ret: isinstance(ret, Notification),
        max_value=45,
        logger=None,
    )
    async def _track_attempt(self, attempt: Attempt, silent=False) -> Any:
        ret = await self.poll_attempt(attempt)
        if not silent:
            if isinstance(ret, Notification):
                delta = datetime.now(timezone.utc) - attempt.started_at
                elapsed = humanize.naturaldelta(
                    delta, minimum_unit="milliseconds"
                )
                if ret.dequeued:
                    msg = "Attempt is running..."
                    details = [f"elapsed={elapsed}"]
                    if ret.relative_gap is not None:
                        details.append(
                            f"gap={format_percent(ret.relative_gap)}"
                        )
                    if ret.cut_count is not None:
                        details.append(f"cuts={ret.cut_count}")
                    if ret.lp_iteration_count is not None:
                        details.append(f"iterations={ret.lp_iteration_count}")
                    if details:
                        msg += f" [{', '.join(details)}]"
                    print(msg)
                else:
                    print(f"Attempt is queued... [elapsed={elapsed}]")
            else:
                delta = ret.reached_at - attempt.started_at
                elapsed = humanize.naturaldelta(
                    delta, minimum_unit="milliseconds"
                )
                details = [f"elapsed={elapsed}"]
                if isinstance(ret, InfeasibleOutcome):
                    print(f"Attempt is infeasible. [{', '.join(details)}]")
                elif isinstance(ret, UnboundedOutcome):
                    print(f"Attempt is unbounded. [{', '.join(details)}]")
                elif isinstance(ret, CancelledOutcome):
                    print(f"Attempt cancelled. [{', '.join(details)}]")
                elif isinstance(ret, FailedOutcome):
                    details.append(f"status={ret.status}")
                    details.append(f"message={ret.message}")
                    print(f"Attempt failed. [{', '.join(details)}]")
                else:
                    adj = "optimal" if ret.is_optimal else "feasible"
                    if ret.objective_value is not None:
                        details.append(f"objective={ret.objective_value}")
                    if ret.relative_gap is not None:
                        details.append(
                            f"gap={format_percent(ret.relative_gap)}"
                        )
                    print(f"Attempt is {adj}. [{', '.join(details)}]")
        return ret

    async def wait_for_outcome(
        self,
        attempt: Attempt,
        silent: bool = False,
        assert_feasible: bool = False,
    ) -> Outcome:
        """Waits for the attempt to complete and returns its outcome."""
        print(f"Tracking attempt... [url={attempt.url}]")
        outcome = await self._track_attempt(attempt, silent=silent)
        if assert_feasible and not isinstance(outcome, FeasibleOutcome):
            raise Exception(f"Unexpected outcome: {outcome}")
        return outcome

    async def fetch_attempt_inputs(self, attempt: Attempt) -> SolveInputs:
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            path=f"/attempts/{attempt.uuid}/inputs",
        ) as res:
            data = res.json_data()
        return SolveInputs(
            outline=attempt.outline,
            raw_parameters=data["parameters"],
            raw_dimensions=data["dimensions"],
        )

    async def fetch_attempt_outputs(self, attempt: Attempt) -> SolveOutputs:
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            path=f"/attempts/{attempt.uuid}/outputs",
        ) as res:
            data = res.json_data()
        return SolveOutputs(
            outline=attempt.outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )


class _SolveInputsBuilder:
    def __init__(self, outline: Outline):
        self._outline = outline
        self._dimensions: Dict[Label, Any] = {}
        self._parameters: Dict[Label, Any] = {}
        self.parameter_entry_count = 0

    def set_dimension(self, label: Label, arg: DimensionArgument) -> None:
        outline = self._outline.dimensions.get(label)
        if not outline:
            raise Exception(f"Unknown dimension: {label}")
        if label in self._dimensions:
            raise Exception(f"Duplicate dimension: {label}")
        items = list(arg)
        self._dimensions[label] = {"label": label, "items": items}

    def set_parameter(self, label: Label, arg: Any) -> None:
        outline = self._outline.parameters.get(label)
        if not outline:
            raise Exception(f"Unknown parameter: {label}")
        if label in self._parameters:
            raise Exception(f"Duplicate parameter: {label}")
        try:
            tensor = Tensor.from_argument(
                arg, len(outline.bindings), outline.is_indicator()
            )
        except Exception as exc:
            raise ValueError(f"Invalid  parameter: {label}") from exc
        self._parameters[label] = {
            "label": label,
            "entries": tensor.entries,
            "defaultValue": tensor.default_value,
        }
        self.parameter_entry_count += len(tensor.entries)

    def build(self) -> SolveInputs:
        missing_labels = set()

        for label in self._outline.parameters:
            if label not in self._parameters:
                missing_labels.add(label)

        if self._dimensions:
            for label in self._outline.dimensions:
                if label not in self._dimensions:
                    missing_labels.add(label)

        if missing_labels:
            raise Exception(f"Missing label(s): {missing_labels}")

        return SolveInputs(
            outline=self._outline,
            raw_parameters=list(self._parameters.values()),
            raw_dimensions=list(self._dimensions.values()) or None,
        )
