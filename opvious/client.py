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
import json
import humanize
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from .common import format_percent, strip_nones
from .data import (
    Attempt,
    CancelledOutcome,
    DimensionArgument,
    FailedOutcome,
    FeasibleOutcome,
    InfeasibleOutcome,
    InputData,
    Inputs,
    Label,
    Notification,
    Outcome,
    Outline,
    OutputData,
    Outputs,
    Relaxation,
    Tensor,
    TensorArgument,
    UnboundedOutcome,
)
from .executors import default_executor, Executor, execute_graphql_query


_DEFAULT_DOMAIN = "beta.opvious.io"


_TOKEN_EVAR = "OPVIOUS_TOKEN"


_DOMAIN_EVAR = "OPVIOUS_DOMAIN"


class Client:
    """Opvious API client"""

    def __init__(self, executor: Executor, hub_url: str):
        self._executor = executor
        self._hub_url = hub_url

    @classmethod
    def from_token(cls, token: str, domain: Optional[str] = None):
        """Creates a client from an API token."""
        return Client(
            executor=default_executor(
                api_url=f"https://api.{domain or _DEFAULT_DOMAIN}",
                authorization=token if " " in token else f"Bearer {token}",
            ),
            hub_url=f"https://hub.{domain}",
        )

    @classmethod
    def from_environment(cls, env=os.environ):
        """Creates a client from environment variables. OPVIOUS_TOKEN should
        contain a valid API token. OPVIOUS_DOMAIN can optionally be set to use
        a custom domain.
        """
        return Client.from_token(
            token=env[_TOKEN_EVAR], domain=env.get(_DOMAIN_EVAR)
        )

    async def solve(
        self,
        sources: Optional[list[str]] = None,
        formulation_name: Optional[str] = None,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        relative_gap_threshold: Optional[float] = None,
        absolute_gap_threshold: Optional[float] = None,
        primal_value_epsilon: Optional[float] = None,
        timeout_millis: Optional[float] = None,
    ) -> Outputs:
        """Solves an optimization problem. See also `start_attempt` for an
        alternative for long-running solves.
        """
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
        builder = _InputDataBuilder(outline=outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        inputs = builder.build()
        solve_res = await self._executor.execute(
            path="/solves/run",
            method="POST",
            body={
                "formulation": formulation,
                "inputs": strip_nones(
                    {
                        "dimensions": inputs.raw_dimensions,
                        "parameters": inputs.raw_parameters,
                    }
                ),
                "options": strip_nones(
                    {
                        "absoluteGapThreshold": absolute_gap_threshold,
                        "relativeGapThreshold": relative_gap_threshold,
                        "timeoutMillis": timeout_millis,
                        "primalValueEpsilon": primal_value_epsilon,
                    }
                ),
            },
        )
        solve_data = solve_res.json_data()
        outcome_data = solve_data["outcome"]
        status = outcome_data["status"]
        reached_at = datetime.now(timezone.utc)
        if status == "INFEASIBLE":
            outcome = InfeasibleOutcome(reached_at)
        elif status == "UNBOUNDED":
            outcome = UnboundedOutcome(reached_at)
        else:
            outcome = FeasibleOutcome(
                reached_at=reached_at,
                is_optimal=status == "OPTIMAL",
                objective_value=outcome_data.get("objectiveValue"),
                relative_gap=outcome_data.get("relativeGap"),
            )
        if not isinstance(outcome, FeasibleOutcome):
            return Outputs(status=status, outcome=outcome)
        outputs_data = solve_data["outputs"]
        return Outputs(
            status=status,
            outcome=outcome,
            data=OutputData(
                outline=outline,
                raw_variables=outputs_data["variables"],
                raw_constraints=outputs_data["constraints"],
            ),
        )

    async def _fetch_sources_outline(self, sources: list[str]) -> Outline:
        outline_res = await self._executor.execute(
            path="/sources/parse",
            method="POST",
            body={"sources": sources, "outline": True},
        )
        outline_data = outline_res.json_data()
        errors = outline_data.get("errors")
        if errors:
            raise Exception(f"Invalid sources: {json.dumps(errors)}")
        return Outline.from_json(outline_data["outline"])

    async def _fetch_formulation_outline(
        self, name: str, tag_name: Optional[str] = None
    ) -> Tuple[Outline, str]:
        data = await execute_graphql_query(
            executor=self._executor,
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
        return [outline, tag["name"]]

    async def assemble_inputs(
        self,
        formulation_name: str,
        tag_name: Optional[str] = None,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
    ) -> Inputs:
        """Assembles and validates inputs for a given formulation. The returned
        object can be used to start an asynchronous solve via `start_attempt`.
        """
        outline, tag = await self._fetch_formulation_outline(
            formulation_name, tag_name
        )
        builder = _InputDataBuilder(outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        return Inputs(
            formulation_name=formulation_name,
            tag_name=tag,
            data=builder.build(),
        )

    async def start_attempt(
        self,
        inputs: Inputs,
        relative_gap_threshold: Optional[float] = None,
        absolute_gap_threshold: Optional[float] = None,
        primal_value_epsilon: Optional[float] = None,
        timeout_millis: Optional[float] = None,
        relaxed_constraints: Union[None, List[Label], Relaxation] = None,
        pinned_variables: Optional[Mapping[Label, TensorArgument]] = None,
    ) -> Attempt:
        """Starts a new asynchronous solve attempt."""
        if relaxed_constraints:
            if isinstance(relaxed_constraints, list):
                data = Relaxation.from_constraint_labels(relaxed_constraints)
            else:
                data = relaxed_constraints
            relaxation = data.to_json()
        else:
            relaxation = None
        pins = []
        if pinned_variables:
            for label, arg in pinned_variables.items():
                var_outline = inputs.data.outline.variables.get(label)
                if not var_outline:
                    raise Exception(f"Unknown variable {label}")
                tensor = Tensor.from_argument(arg, var_outline.is_indicator())
                if tensor.default_value:
                    raise Exception("Pinned variables may not have defaults")
                pins.append({"label": label, "entries": tensor.entries})
        result = await self._executor.execute(
            path="/attempts/start",
            method="POST",
            body={
                "formulationName": inputs.formulation_name,
                "specificationTagName": inputs.tag_name,
                "inputs": strip_nones(
                    {
                        "dimensions": inputs.data.raw_dimensions,
                        "parameters": inputs.data.raw_parameters,
                        "pinnedVariables": pins,
                    }
                ),
                "options": strip_nones(
                    {
                        "absoluteGapThreshold": absolute_gap_threshold,
                        "relativeGapThreshold": relative_gap_threshold,
                        "timeoutMillis": timeout_millis,
                        "primalValueEpsilon": primal_value_epsilon,
                        "relaxation": relaxation,
                    }
                ),
            },
        )
        uuid = result.json_data()["uuid"]
        return Attempt(
            uuid=uuid,
            started_at=datetime.now(timezone.utc),
            outline=inputs.data.outline,
            url=self._attempt_url(uuid),
        )

    async def load_attempt(self, uuid: str) -> Optional[Attempt]:
        """Loads an existing attempt from its UUID."""
        data = await execute_graphql_query(
            executor=self._executor,
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
        data = await execute_graphql_query(
            executor=self._executor,
            query="@CancelAttempt",
            variables={"uuid": uuid},
        )
        return bool(data["cancelAttempt"])

    async def poll_attempt(
        self, attempt: Attempt
    ) -> Union[Notification, Outcome]:
        """Polls an attempt for its outcome or latest progress notification."""
        data = await execute_graphql_query(
            executor=self._executor,
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
            return CancelledOutcome(reached_at)
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

    async def fetch_input_data(self, attempt: Attempt) -> InputData:
        res = await self._executor.execute(f"/attempts/{attempt.uuid}/inputs")
        data = res.json_data()
        return InputData(
            outline=attempt.outline,
            raw_parameters=data["parameters"],
            raw_dimensions=data["dimensions"],
        )

    async def fetch_output_data(self, attempt: Attempt) -> OutputData:
        res = await self._executor.execute(f"/attempts/{attempt.uuid}/outputs")
        data = res.json_data()
        return OutputData(
            outline=attempt.outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )


class _InputDataBuilder:
    def __init__(self, outline: Outline):
        self._outline = outline
        self._dimensions: Dict[Label, Any] = {}
        self._parameters: Dict[Label, Any] = {}

    def set_dimension(self, label: Label, arg: DimensionArgument) -> None:
        outline = self._outline.dimensions.get(label)
        if not outline:
            raise Exception(f"Unknown dimension: {label}")
        self._dimensions[label] = {
            "label": label,
            "items": list(arg),
        }

    def set_parameter(self, label: Label, arg: Any) -> None:
        outline = self._outline.parameters.get(label)
        if not outline:
            raise Exception(f"Unknown parameter: {label}")
        tensor = Tensor.from_argument(arg, outline.is_indicator())
        self._parameters[label] = {
            "label": label,
            "entries": tensor.entries,
            "defaultValue": tensor.default_value,
        }

    def build(self) -> Inputs:
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

        return InputData(
            outline=self._outline,
            raw_parameters=list(self._parameters.values()),
            raw_dimensions=list(self._dimensions.values()) or None,
        )
