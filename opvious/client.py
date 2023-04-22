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

import asyncio
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

from .common import format_percent, is_url, strip_nones
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
    Relaxation,
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    solve_options_to_json,
    SolveSummary,
)
from .data.tensors import DimensionArgument, Tensor, TensorArgument
from .executors import (
    default_executor,
    Executor,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
)
from .specifications import FormulationSpecification, Specification


_logger = logging.getLogger(__name__)


_DEFAULT_DOMAIN = "beta.opvious.io"


class ClientSettings(enum.Enum):
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
                root_url=api_url,
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
        token = env.get(ClientSettings.TOKEN.value, "")
        if not token and require_authenticated:
            raise Exception(
                f"Missing or empty {ClientSettings.TOKEN.value} environment "
                + "variable"
            )
        return Client.from_token(
            token=token, domain=env.get(ClientSettings.DOMAIN.value)
        )

    @property
    def authenticated(self):
        """
        Returns true if the client was created with a non-empty API token.
        """
        return self._executor.authenticated

    async def inspect_solve_instructions(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        relaxation: Optional[Relaxation] = None,
        options: Optional[SolveOptions] = None,
    ) -> str:
        """Inspects an optimization problem's instructions, returning its
        LP formatted representation.
        """
        body, _outline = await self._assemble_solve_request(
            specification=specification,
            parameters=parameters,
            dimensions=dimensions,
            relaxation=relaxation,
            options=options,
        )
        async with self._executor.execute(
            result_type=PlainTextExecutorResult,
            url="/solves/inspect/instructions",
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

    async def run_solve(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        relaxation: Optional[Relaxation] = None,
        options: Optional[SolveOptions] = None,
        assert_feasible=False,
        prefer_streaming=True,
    ) -> SolveResponse:
        """Solves an optimization problem. See also `start_attempt` for an
        alternative for long-running solves.
        """
        body, outline = await self._assemble_solve_request(
            specification=specification,
            parameters=parameters,
            dimensions=dimensions,
            relaxation=relaxation,
            options=options,
        )

        if prefer_streaming and self._executor.supports_streaming:
            summary = None
            response_json = None
            async with self._executor.execute(
                result_type=JsonSeqExecutorResult,
                url="/solves/run",
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
                        summary = SolveSummary.from_json(data["summary"])
                        _logger.info(
                            "Solving problem... [columns=%s, rows=%s]",
                            summary.column_count,
                            summary.row_count,
                        )
                    elif kind == "solving":
                        progress = data["progress"]
                        iter_count = progress.get("lpIterationCount")
                        gap = progress.get("relativeGap")
                        if iter_count is not None:
                            _logger.info(
                                "Solve in progress... [iters=%s, gap=%s]",
                                iter_count,
                                "n/a" if gap is None else format_percent(gap),
                            )
                    elif kind == "solved":
                        _logger.debug("Downloaded outputs.")
                        response_json = data
                    elif kind == "error":
                        message = data["error"]["message"]
                        raise Exception(f"Solve failed: {message}")
                    else:
                        raise Exception(
                            f"Unexpected response: {json.dumps(data)}"
                        )
            if not summary or not response_json:
                raise Exception("Streaming solve terminated early")
            response = SolveResponse.from_json(
                outline=outline,
                response_json=response_json,
                summary=summary,
            )
        else:
            async with self._executor.execute(
                result_type=JsonExecutorResult,
                url="/solves/run",
                method="POST",
                json_data=body,
            ) as res:
                response = SolveResponse.from_json(
                    outline=outline,
                    response_json=res.json_data(),
                )

        if assert_feasible and not isinstance(
            response.outcome, FeasibleOutcome
        ):
            raise UnexpectedOutcomeError(response.outcome)
        return response

    async def _assemble_solve_request(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        options: Optional[SolveOptions] = None,
        relaxation: Union[None, Relaxation] = None,
    ) -> Tuple[Any, Outline]:
        # First we fetch the outline to validate/coerce inputs later on
        if isinstance(specification, FormulationSpecification):
            outline, tag_name = await self._fetch_formulation_outline(
                specification
            )
            formulation = strip_nones(
                {
                    "name": specification.formulation_name,
                    "specificationTagName": tag_name,
                }
            )
        else:
            sources = await specification.fetch_sources(self._executor)
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

    async def _resolving_sources(self, sources: list[str]) -> list[str]:
        async def _resolve(source: str) -> str:
            if not is_url(source):
                return source
            _logger.debug("Resolving source URL... [url=%s]", source)
            try:
                async with self._executor.execute(
                    result_type=PlainTextExecutorResult,
                    url=source,
                ) as res:
                    return await res.text(assert_status=200)
            except Exception as exc:
                raise Exception(
                    f"Unable to read source from {source}"
                ) from exc

        tasks = [_resolve(source) for source in sources]
        return await asyncio.gather(*tasks)

    async def _fetch_sources_outline(self, sources: list[str]) -> Outline:
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url="/sources/parse",
            method="POST",
            json_data={"sources": sources, "outline": True},
        ) as res:
            outline_data = res.json_data()
        errors = outline_data.get("errors")
        if errors:
            raise Exception(f"Invalid sources: {json.dumps(errors)}")
        return Outline.from_json(outline_data["outline"])

    async def _fetch_formulation_outline(
        self, specification: FormulationSpecification
    ) -> Tuple[Outline, str]:
        data = await self._executor.execute_graphql_query(
            query="@FetchOutline",
            variables={
                "formulationName": specification.formulation_name,
                "tagName": specification.tag_name,
            },
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
        specification: Union[str, FormulationSpecification],
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
    ) -> AttemptRequest:
        """Assembles and validates inputs for a given formulation. The returned
        object can be used to start an asynchronous solve via `start_attempt`.
        """
        if isinstance(specification, str):
            specification = FormulationSpecification(
                formulation_name=specification
            )
        outline, tag_name = await self._fetch_formulation_outline(
            specification
        )
        builder = _SolveInputsBuilder(outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        return AttemptRequest(
            formulation_name=specification.formulation_name,
            specification_tag_name=tag_name,
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
            url="/attempts/start",
            method="POST",
            json_data={
                "formulationName": request.formulation_name,
                "specificationTagName": request.specification_tag_name,
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
        if status == "CANCELLED":
            return cast(Outcome, CancelledOutcome())
        if status == "INFEASIBLE":
            return InfeasibleOutcome()
        if status == "UNBOUNDED":
            return UnboundedOutcome()
        outcome = attempt_data["outcome"]
        if status == "ERRORED":
            return FailedOutcome.from_graphql(outcome)
        if status == "FEASIBLE" or status == "OPTIMAL":
            return FeasibleOutcome.from_graphql(outcome)
        raise Exception(f"Unexpected status {status}")

    @backoff.on_predicate(
        backoff.fibo,
        lambda ret: ret is None,
        max_value=45,
        logger=None,
    )
    async def _track_attempt(self, attempt: Attempt) -> Optional[Outcome]:
        ret = await self.poll_attempt(attempt)
        if isinstance(ret, Notification):
            delta = datetime.now(timezone.utc) - attempt.started_at
            elapsed = humanize.naturaldelta(delta, minimum_unit="milliseconds")
            if ret.dequeued:
                details = [f"elapsed={elapsed}"]
                if ret.relative_gap is not None:
                    details.append(f"gap={format_percent(ret.relative_gap)}")
                if ret.cut_count is not None:
                    details.append(f"cuts={ret.cut_count}")
                if ret.lp_iteration_count is not None:
                    details.append(f"iterations={ret.lp_iteration_count}")
                _logger.info("Attempt is running... [%s]", ", ".join(details))
            else:
                _logger.info("Attempt is queued... [elapsed=%s]", elapsed)
            return None
        return ret

    async def wait_for_outcome(
        self,
        attempt: Attempt,
        assert_feasible=False,
    ) -> Outcome:
        """
        Waits for the attempt to complete and returns its outcome. Enable INFO
        logging to view progress messages.
        """
        outcome = await self._track_attempt(attempt)
        if not outcome:
            raise Exception("Missing outcome")
        if assert_feasible and not isinstance(outcome, FeasibleOutcome):
            raise UnexpectedOutcomeError(outcome)
        return outcome

    async def fetch_attempt_inputs(self, attempt: Attempt) -> SolveInputs:
        """Retrieves an attempt's inputs."""
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url=f"/attempts/{attempt.uuid}/inputs",
        ) as res:
            data = res.json_data()
        return SolveInputs(
            outline=attempt.outline,
            raw_parameters=data["parameters"],
            raw_dimensions=data["dimensions"],
        )

    async def fetch_attempt_outputs(self, attempt: Attempt) -> SolveOutputs:
        """Retrieves a successful attempt's outputs."""
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url=f"/attempts/{attempt.uuid}/outputs",
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
