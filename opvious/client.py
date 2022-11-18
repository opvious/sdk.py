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
import humanize
import pandas as pd
from typing import Any, Dict, Mapping, Optional, Union

from .data import (
    Attempt,
    CancelledOutcome,
    InfeasibleOutcome,
    UnboundedOutcome,
    FailedOutcome,
    FeasibleOutcome,
    is_value,
    Inputs,
    Label,
    Notification,
    Outcome,
    Outline,
    ParameterArgument,
    DimensionArgument,
    RelaxedConstraint,
)
from .executors import aiohttp_executor


DEFAULT_API_URL = "https://api.opvious.io"


DEFAULT_HUB_URL = "https://hub.opvious.io"


_DEFAULT_PENALTY = "TOTAL_DEVIATION"


class Client:
    """Opvious API client"""

    def __init__(
        self,
        token,
        api_url=None,
        hub_url=None,
    ):
        self._api_url = api_url or DEFAULT_API_URL
        self._hub_url = hub_url or DEFAULT_HUB_URL
        auth = token if " " in token else f"Bearer {token}"
        self._executor = aiohttp_executor(self._api_url, auth)

    async def assemble_inputs(
        self,
        formulation_name: str,
        parameters: Optional[Mapping[Label, ParameterArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        tag_name: Optional[str] = None,
        infer_dimensions: bool = False,
    ) -> Inputs:
        """ "Assembles and validates inputs."""
        data = await self._executor.execute(
            "@FetchOutline",
            {
                "formulationName": formulation_name,
                "tagName": tag_name,
            },
        )
        formulation = data.get("formulation")
        if not formulation:
            raise Exception("No matching formulation found")
        tag = formulation.get("tag")
        if not tag:
            raise Exception("No matching specification found")
        spec = tag["specification"]
        builder = _InputsBuilder(
            formulation_name=formulation_name,
            tag_name=tag["name"],
            outline=Outline.from_graphql(spec["outline"]),
        )
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        return builder.build(infer_dimensions=infer_dimensions)

    async def start_attempt(
        self,
        inputs: Inputs,
        relative_gap_threshold: Optional[float] = None,
        absolute_gap_threshold: Optional[float] = None,
        primal_value_epsilon: Optional[float] = None,
        solve_timeout_millis: Optional[float] = None,
        relaxed_constraints: Optional[
            list[Union[Label, RelaxedConstraint]]
        ] = None,
        # TODO: pinned variables
    ) -> Attempt:
        """Starts a new attempt."""
        if relaxed_constraints:
            relaxation = {
                "penalty": _DEFAULT_PENALTY,
                "constraints": [
                    c.to_graphql()
                    if isinstance(c, RelaxedConstraint)
                    else {"label": c}
                    for c in relaxed_constraints
                ],
            }
        else:
            relaxation = None
        data = await self._executor.execute(
            "@StartAttempt",
            {
                "input": {
                    "formulationName": inputs.formulation_name,
                    "specificationTagName": inputs.tag_name,
                    "dimensions": inputs.dimensions,
                    "parameters": inputs.parameters,
                    "absoluteGapThreshold": absolute_gap_threshold,
                    "relativeGapThreshold": relative_gap_threshold,
                    "solveTimeoutMillis": solve_timeout_millis,
                    "primalValueEpsilon": primal_value_epsilon,
                    "relaxation": relaxation,
                }
            },
        )
        attempt_data = data["startAttempt"]
        return Attempt.from_graphql(
            data=attempt_data,
            outline=inputs.outline,
            url=self._attempt_url(attempt_data["uuid"]),
        )

    async def load_attempt(self, uuid: str) -> Optional[Attempt]:
        """Loads an existing attempt from its UUID."""
        data = await self._executor.execute("@FetchAttempt", {"uuid": uuid})
        attempt = data["attempt"]
        if not attempt:
            return None
        return Attempt.from_graphql(
            data=attempt,
            outline=Outline.from_graphql(attempt["outline"]),
            url=self._attempt_url(uuid),
        )

    def _attempt_url(self, uuid) -> str:
        return self._hub_url + "/attempts/" + uuid

    async def cancel_attempt(self, uuid: str) -> bool:
        """Cancels a running attempt."""
        data = await self._executor.execute("@CancelAttempt", {"uuid": uuid})
        return bool(data["cancelAttempt"])

    async def poll_attempt(
        self, attempt: Attempt
    ) -> Union[Notification, Outcome]:
        """Polls an attempt for its outcome or latest progress notification."""
        data = await self._executor.execute(
            "@PollAttempt",
            {
                "uuid": attempt.uuid,
            },
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
        if status == "FAILED":
            return FailedOutcome.from_graphql(reached_at, outcome)
        return FeasibleOutcome.from_graphql(reached_at, outcome)

    @backoff.on_predicate(
        backoff.fibo, lambda ret: isinstance(ret, Notification), max_value=45
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
                        details.append(f"gap={_percent(ret.relative_gap)}")
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
                    details.append(f"objective={ret.objective_value}")
                    if ret.relative_gap is not None:
                        details.append(f"gap={_percent(ret.relative_gap)}")
                    print(f"Attempt is {adj}. [{', '.join(details)}]")
        return ret

    async def wait_for_outcome(
        self, attempt: Attempt, silent=False
    ) -> Outcome:
        """Waits for the attempt to complete and returns its outcome."""
        print(f"Tracking attempt... [url={attempt.url}]")
        return await self._track_attempt(attempt, silent=silent)

    async def _fetch_inputs(self, uuid: str) -> Any:
        data = await self._executor.execute(
            "@FetchAttemptInputs", {"uuid": uuid}
        )
        return data["attempt"]

    async def fetch_dimension(
        self, attempt: Attempt, label: Label
    ) -> pd.Series:
        inputs = await self._fetch_inputs(attempt.uuid)
        for dim in inputs["dimensions"]:
            if dim["label"] == label:
                return pd.Index(dim["items"])
        raise Exception(f"Unknown dimension: {label}")

    async def fetch_parameter(
        self, attempt: Attempt, label: Label
    ) -> pd.Series:
        inputs = await self._fetch_inputs(attempt.uuid)
        for param in inputs["parameters"]:
            if param["label"] == label:
                entries = param["entries"]
                return pd.Series(
                    data=(e["value"] for e in entries),
                    index=pd.Index(tuple(e["key"]) for e in entries),
                )
        raise Exception(f"Unknown parameter: {label}")

    async def _fetch_outputs(self, uuid: str):
        data = await self._executor.execute(
            "@FetchAttemptOutputs", {"uuid": uuid}
        )
        outcome = data["attempt"].get("outcome")
        if not outcome or outcome["__typename"] != "FeasibleOutcome":
            raise Exception("Missing or non-feasible attempt outcome")
        return outcome

    async def fetch_variable(
        self, attempt: Attempt, label: Label
    ) -> pd.DataFrame:
        outputs = await self._fetch_outputs(attempt.uuid)
        for var in outputs["variables"]:
            if var["label"] == label:
                entries = var["entries"]
                df = pd.DataFrame(
                    data=(
                        {"value": e["value"], "dual_value": e["dualValue"]}
                        for e in entries
                    ),
                    index=pd.Index(tuple(e["key"]) for e in entries),
                )
                return df.dropna(axis=1)
        raise Exception(f"Unknown variable {label}")

    async def fetch_constraint(
        self, attempt: Attempt, label: Label
    ) -> pd.DataFrame:
        outputs = await self._fetch_outputs(attempt.uuid)
        for var in outputs["constraints"]:
            if var["label"] == label:
                entries = var["entries"]
                df = pd.DataFrame(
                    data=(
                        {"slack": e["value"], "dual_value": e["dualValue"]}
                        for e in entries
                    ),
                    index=pd.Index(tuple(e["key"]) for e in entries),
                )
                return df.dropna(axis=1)
        raise Exception(f"Unknown constraint {label}")


class _InputsBuilder:
    def __init__(self, formulation_name: str, tag_name: str, outline: Outline):
        self._formulation_name = formulation_name
        self._tag_name = tag_name
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
        if isinstance(arg, tuple):
            data, default_value = arg
        else:
            data = arg
            default_value = 0
        is_indic = outline.is_indicator()
        if (
            is_indic
            and isinstance(data, pd.Series)
            and not pd.api.types.is_numeric_dtype(data)
        ):
            data = data.reset_index()
        if is_indic and isinstance(data, pd.DataFrame):
            entries = [
                {"key": key, "value": 1}
                for key in data.itertuples(index=False, name=None)
            ]
        elif is_indic and not hasattr(data, "items"):
            entries = [{"key": key, "value": 1} for key in data]
        else:
            if is_value(data):
                entries = [{"key": (), "value": data}]
            else:
                entries = [
                    {"key": _keyify(key), "value": value}
                    for key, value in data.items()
                ]
        self._parameters[label] = {
            "label": label,
            "entries": entries,
            "defaultValue": default_value,
        }

    def build(self, infer_dimensions=False) -> Inputs:
        missing_labels = set()

        for label in self._outline.parameters:
            if label not in self._parameters:
                missing_labels.add(label)

        dimensions = dict(self._dimensions)
        for label in self._outline.dimensions:
            if label in self._dimensions:
                continue
            if not infer_dimensions:
                missing_labels.add(label)
                continue
            if missing_labels:
                continue
            items = set()
            for outline in self._outline.parameters.values():
                for i, binding in enumerate(outline.bindings):
                    if binding.dimension_label != label:
                        continue
                    entries = self._parameters[outline.label]["entries"]
                    for entry in entries:
                        items.add(entry["key"][i])
            dimensions[label] = {"label": label, "items": list(items)}

        if missing_labels:
            raise Exception(f"Missing label(s): {missing_labels}")

        return Inputs(
            formulation_name=self._formulation_name,
            tag_name=self._tag_name,
            outline=self._outline,
            dimensions=list(dimensions.values()),
            parameters=list(self._parameters.values()),
        )


def _keyify(key):
    return tuple(key) if isinstance(key, (list, tuple)) else (key,)


def _percent(val):
    if val == "Infinity":
        return "inf"
    return f"{int(val * 10_000) / 100}%"
