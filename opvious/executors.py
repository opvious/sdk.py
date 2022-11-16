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

import json
import logging
import sys
from typing import Any, Mapping, Protocol


logger = logging.getLogger(__name__)


class Executor(Protocol):
    async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
        pass


_GRAPHQL_ENDPOINT = '/graphql'


class ApiError(Exception):
    def __init__(self, status, errors=None, extensions=None):
        super().__init__(f'API call failed with status {status}')
        self.status = status
        self.errors = errors
        self.extensions = extensions


def _extract_response_data(status: int, body: Any) -> Any:
    errors = body.get('errors')
    if status != 200 or errors:
        raise ApiError(status, errors, body.get('extensions'))
    return body['data']


def aiohttp_executor(url: str, auth: str) -> Executor:
    import aiohttp

    headers = {
        'authorization': auth,
        'opvious-client': 'Python SDK (aiohttp)',
    }

    class AiohttpExecutor:
        """`aiohttp`-powered GraphQL executor"""

        async def execute(self, query: str, variables: Mapping[str, Any]) -> Any:
            data = {'query': query, 'variables': variables}
            async with aiohttp.ClientSession(headers=headers) as session:
                endpoint = url + _GRAPHQL_ENDPOINT
                async with session.post(endpoint, json=data) as res:
                    logger.info('Trace: %s', res.headers.get('opvious-trace'))
                    body = await res.json()
                    return _extract_response_data(res.status, body)

    return AiohttpExecutor()
