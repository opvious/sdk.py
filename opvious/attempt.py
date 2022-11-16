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

import pandas as pd
import time
from typing import Any, Dict, Optional

from .data import (
    FailedOutcome,
    Inputs,
    Label,
    InfeasibleOutcome,
    UnboundedOutcome,
    FeasibleOutcome,
    Outcome,
)
from .executors import Executor


class Attempt:
    def __init__(self, uuid: str, executor: Executor):
        self.uuid = uuid
        self._executor = executor

    async def fetch_outcome(self) -> Optional[Outcome]:
        data = await self._executor.execute(
            "@PollAttempt",
            {
                "uuid": self.uuid,
            },
        )
        attempt = data["attempt"]
        status = attempt["status"]
        if status == "PENDING":
            return None
        if status == "INFEASIBLE":
            return InfeasibleOutcome()
        if status == "UNBOUNDED":
            return UnboundedOutcome()
        outcome = attempt["outcome"]
        if status == "FAILED":
            failure = outcome["failure"]
            return FailedOutcome(
                status=failure["status"],
                message=failure["message"],
                code=failure.get("code"),
                tags=failure.get("tags"),
            )
        return FeasibleOutcome(
            is_optimal=outcome["isOptimal"],
            objective_value=outcome["objectiveValue"],
            relative_gap=outcome.get("relativeGap"),
        )

    async def wait_for_outcome(self) -> Outcome:
        while True:
            time.sleep(1)
            outcome = await self.fetch_outcome()
            if outcome:
                return outcome

    async def _fetch_inputs(self):
        data = await self._execute("@FetchAttemptInputs", {"uuid": self.uuid})
        return data["attempt"]

    async def load_dimension(self, label: Label) -> pd.Index:
        inputs = self._fetch_inputs()
        for dim in inputs["dimensions"]:
            if dim["label"] == label:
                return pd.Index(dim["items"])
        raise Exception(f"Unknown dimension {label}")

    async def load_parameter(self, label: Label) -> pd.Series:
        inputs = self._fetch_inputs()
        for param in inputs["parameters"]:
            if param["label"] == label:
                entries = param["entries"]
                return pd.Series(
                    data=(e["value"] for e in entries),
                    index=pd.Index(e["key"] for e in entries),
                )
        raise Exception(f"Unknown parameter {label}")

    async def load_variable_result(self, label: Label) -> pd.Series:
        data = await self._executor.execute(
            "@FetchAttemptOutputs",
            {
                "uuid": self.uuid,
            },
        )
        outcome = data["attempt"].get("outcome")
        if not outcome or outcome["__typename"] != "FeasibleOutcome":
            raise Exception("Missing or non-feasible attempt outcome")
        for var in outcome["variables"]:
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


class InputsBuilder:
    def __init__(self, formulation_name: str, tag_name: str, outline: Any):
        self.formulation_name = formulation_name
        self.tag_name = tag_name
        self._outline = outline
        self._parameters: Dict[Label, Any] = {}
        self._dimensions: Dict[Label, Any] = {}

    @property
    def dimension_labels(self) -> list[Label]:
        return [p["label"] for p in self._outline["dimensions"]]

    @property
    def parameter_labels(self) -> list[Label]:
        return [p["label"] for p in self._outline["parameters"]]

    def _find(self, key: str, label: Label) -> Any:
        outline = self._outline
        return next((e for e in outline[key] if e["label"] == label), None)

    def set(self, label: Label, data: Any, default_value=None) -> None:
        """Set a parameter or dimension"""
        param_outline = self._find("parameters", label)
        if param_outline:
            is_indic = _is_indicator(param_outline)
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
            else:
                if isinstance(data, (float, int)):
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
            return
        dim = self._find("dimensions", label)
        if dim:
            self._dimensions[label] = {
                "label": label,
                "items": list(data),
            }
            return
        raise Exception(f"Unknown label {label}")

    def build(self, infer_dimensions=False) -> Inputs:
        missing_labels = set()

        for outline in self._outline["parameters"]:
            label = outline["label"]
            if label not in self._parameters:
                missing_labels.add(label)

        dimensions = dict(self._dimensions)
        for dim_outline in self._outline["dimensions"]:
            label = dim_outline["label"]
            if label in self._dimensions:
                continue
            if not infer_dimensions:
                missing_labels.add(label)
                continue
            if missing_labels:
                continue
            items = set()
            for outline in self._outline["parameters"]:
                for i, binding in enumerate(outline["bindings"]):
                    if binding.get("dimensionLabel") != label:
                        continue
                    entries = self._parameters[outline["label"]]["entries"]
                    for entry in entries:
                        items.add(entry["key"][i])
            dimensions[label] = {"label": label, "items": list(items)}

        if missing_labels:
            raise Exception(f"Missing label(s): {missing_labels}")

        return Inputs(
            formulation_name=self.formulation_name,
            tag_name=self.tag_name,
            dimensions=list(dimensions.values()),
            parameters=list(self._parameters.values()),
        )


def _is_indicator(outline):
    return (
        outline["isIntegral"]
        and outline["lowerBound"] == 0
        and outline["upperBound"] == 1
    )


def _keyify(key):
    return tuple(key) if isinstance(key, (list, tuple)) else (key,)
