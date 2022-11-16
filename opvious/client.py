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

from typing import Optional

from .attempt import Attempt, InputsBuilder, Inputs
from .executors import aiohttp_executor


DEFAULT_API_URL = "https://api.opvious.io"


class Client:
    """Opvious API client"""

    def __init__(self, token, api_url=None):
        self.api_url = api_url or DEFAULT_API_URL
        auth = token if " " in token else f"Bearer {token}"
        self._executor = aiohttp_executor(self.api_url, auth)

    async def create_inputs_builder(
        self, formulation_name: str, tag_name=None
    ) -> InputsBuilder:
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
        return InputsBuilder(formulation_name, tag["name"], spec["outline"])

    async def start_attempt(
        self,
        inputs: Inputs,
        relative_gap=None,
        absolute_gap=None,
        primal_value_epsilon=None,
        solve_timeout_millis=None,
    ) -> Attempt:
        data = await self._executor.execute(
            "@StartAttempt",
            {
                "input": {
                    "formulationName": inputs.formulation_name,
                    "specificationTagName": inputs.tag_name,
                    "dimensions": inputs.dimensions,
                    "parameters": inputs.parameters,
                    "absoluteGapThreshold": absolute_gap,
                    "relativeGapThreshold": relative_gap,
                    "solveTimeoutMillis": solve_timeout_millis,
                    "primalValueEpsilon": primal_value_epsilon,
                }
            },
        )
        uuid = data["startAttempt"]["uuid"]
        return Attempt(uuid, self._executor)

    async def load_attempt(self, uuid: str) -> Optional[Attempt]:
        data = await self._executor.execute("@FetchAttempt", {"uuid": uuid})
        return Attempt(uuid, self._executor) if data["attempt"] else None

    async def cancel_attempt(self, uuid: str) -> bool:
        data = await self._executor.execute("@CancelAttempt", {"uuid": uuid})
        return bool(data["cancelAttempt"])
