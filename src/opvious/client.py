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
import numbers
import sys
import time

from .common import preparing_keys
from .data import *

_DEFAULT_API_URL = 'https://api.opvious.dev'
_GRAPHQL_ENDPOINT = '/graphql'
_SHARED_FORMULATION_ENDPOINT = '/shared/formulations/'

def is_using_pyodide():
  # https://pyodide.org/en/stable/usage/faq.html#how-to-detect-that-code-is-run-with-pyodide
  return 'pyodide' in sys.modules

def pyodide_executor(url, auth):
  from pyodide.http import pyfetch

  class PyodideExecutor:
    """`pyodide`-powered GraphQL executor"""

    async def execute(self, query, variables):
      res = await pyfetch(
        url + _GRAPHQL_ENDPOINT,
        method='POST',
        headers={
          'authorization': auth,
          'content-type': 'application/json',
          'opvious-client': 'Python SDK (pyodide)',
        },
        body=json.dumps({'query': query, 'variables': variables})
      )
      body = await res.js_response.text()
      headers = res.js_response.headers # TODO: This doesn't work.
      return (headers.get('operation'), json.loads(body))

  return PyodideExecutor()

def aiohttp_executor(url, auth):
  import aiohttp

  headers = {
    'authorization': auth,
    'opvious-client': 'Python SDK (aiohttp)',
  }

  class AiohttpExecutor:
    """`aiohttp`-powered GraphQL executor"""

    async def execute(self, query, variables):
      data = {'query': query, 'variables': variables}
      async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(url + _GRAPHQL_ENDPOINT, json=data) as res:
          data = await res.json()
          return (res.headers.get('operation'), data)

  return AiohttpExecutor()

class Client:
  """Opvious API client"""

  def __init__(self, access_token, api_url=None):
    self.api_url = api_url or _DEFAULT_API_URL
    self.authorization_header = f'Bearer {access_token}'
    if is_using_pyodide():
      self._executor = pyodide_executor(self.api_url, self.authorization_header)
    else:
      self._executor = aiohttp_executor(self.api_url, self.authorization_header)
    self.latest_operation = None

  async def _execute(self, query, variables):
    op, res = await self._executor.execute(query, variables)
    self.latest_operation = op
    if res.get('errors'):
      raise Exception(f"Operation {op} failed: {json.dumps(res['errors'])}")
    return res['data']

  async def extract_definitions(self, sources: list[str]):
    data = await self._execute('@ExtractDefinitions', {'sources': sources})
    slices = data['extractDefinitions']['slices']
    defs = []
    for s in slices:
      if s['__typename'] != 'ValidSourceSlice':
        raise Exception(f"Invalid source: {json.dumps(slices)}")
      defs.append(s['definition'])
    return defs

  async def compile_specification(self, definitions: list[Definition]):
    data = await self._execute('@CompileSpecification', {
      'definitions': definitions,
    })
    return data['compileSpecification']['assembly']

  async def get_formulation(self, name: str) -> Formulation:
    data = await self._execute('@FetchFormulation', {'name': name})
    formulation = data['formulation']
    return Formulation(**preparing_keys(formulation)) if formulation else None

  async def update_formulation(
    self,
    name: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None
  ) -> None:
    await self._execute('@UpdateFormulation', {
      'input': {
        'name': name,
        'patch': {
          'displayName': display_name,
          'description': description,
          'url': url,
        },
      }
    })

  async def delete_formulation(self, name: str) -> None:
    await self._execute('@DeleteFormulation', {'name': name})

  async def register_specification(
    self,
    formulation_name,
    definitions,
    tag_names=None
  ):
    data = await self._execute('@RegisterSpecification', {
      'input': {
        'formulationName': formulation_name,
        'definitions': definitions,
        'tagNames': tag_names,
      }
    })
    return data['registerSpecification']['assembly']

  async def share_formulation(self, name: str, tag_name: str) -> str:
    data = await self._execute('@StartSharingFormulation', {
      'input': {
        'name': name,
        'tagName': tag_name,
      },
    })
    slug = data['startSharingFormulation']['sharedVia']
    return f'{self.api_url}{_SHARED_FORMULATION_ENDPOINT}{slug}'

  async def unshare_formulation(self, name: str, tag_names = None) -> str:
    await self._execute('@StopSharingFormulation', {
      'input': {
        'name': name,
        'tagNames': tag_names,
      },
    })

  async def start_attempt(
    self,
    formulation_name,
    tag_name=None,
    dimensions=None,
    parameters=None,
    relative_gap=None,
    absolute_gap=None,
    primal_value_epsilon=None,
    solve_timeout_millis=None
  ) -> str:
    start_data = await self._execute('@StartAttempt', {
      'input': {
        'formulationName': formulation_name,
        'tagName': tag_name,
        'dimensions': [d.to_input() for d in dimensions] if dimensions else [],
        'parameters': [p.to_input() for p in parameters] if parameters else [],
        'options': {
          'absoluteGap': absolute_gap,
          'relativeGap': relative_gap,
          'solveTimeoutMillis': solve_timeout_millis,
          'primalValueEpsilon': primal_value_epsilon,
        },
      }
    })
    return start_data['startAttempt']['uuid']

  async def poll_attempt_outcome(self, uuid) -> Outcome:
    while True:
      time.sleep(1)
      poll_data = await self._execute('@PollAttempt', {'uuid': uuid})
      attempt = poll_data['attempt']
      status = attempt['status']
      if status == 'PENDING':
        continue
      if status == 'FAILED':
        return _failed_outcome(attempt['outcome'])
      if status == 'INFEASIBLE':
        return InfeasibleOutcome()
      if status == 'UNBOUNDED':
        return UnboundedOutcome()
      return _feasible_outcome(attempt['outcome'], attempt['outputs'])

  async def get_attempt_parameters(self, uuid: str) -> list[Parameter]:
    data = await self._execute('@FetchAttemptInputs', {'uuid': uuid})
    return [
      Parameter(
        p['label'],
        [ParameterEntry(e['key'], e['value']) for e in p['entries']],
        p['defaultValue']
      )
      for p in data['attempt']['inputs']['parameters']
    ]

  async def get_attempt_dimensions(self, uuid: str) -> list[Dimension]:
    data = await self._execute('@FetchAttemptInputs', {'uuid': uuid})
    return [
      Dimension(d['label'], d['items'])
      for d in data['attempt']['inputs']['dimensions']
    ]

def _failed_outcome(data):
  failure = data['failure']
  return FailedOutcome(**preparing_keys(failure))

def _feasible_outcome(outcome, outputs):
  return FeasibleOutcome(
    variable_results=[_result(r) for r in outputs['variableResults']],
    constraint_results=[_result(r) for r in outputs['constraintResults']],
    **preparing_keys(outcome)
  )

def _result(data):
  label = data['label']
  entries = data['entries']
  if len(entries) == 1 and not entries[0]['key']:
    return ScalarResult(label=label, value=entries[0]['primalValue'])
  value = {}
  for entry in entries:
    key = entry['key']
    value[tuple(key) if len(key) > 1 else key[0]] = entry['primalValue']
  return IndexedResult(label=label, value=value)
