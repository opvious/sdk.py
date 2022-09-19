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

from .data import *

API_URL = 'https://api.opvious.io'
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
      return (headers.get('opvious-trace'), json.loads(body))

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
          return (res.headers.get('opvious-trace'), data)

  return AiohttpExecutor()

class Client:
  """Opvious API client"""

  def __init__(self, access_token, api_url=None):
    self.api_url = api_url or API_URL
    self.authorization_header = f'Bearer {access_token}'
    if is_using_pyodide():
      self._executor = pyodide_executor(self.api_url, self.authorization_header)
    else:
      self._executor = aiohttp_executor(self.api_url, self.authorization_header)
    self.latest_trace = None

  async def _execute(self, query, variables):
    tid, res = await self._executor.execute(query, variables)
    self.latest_trace = tid
    if res.get('errors'):
      raise Exception(f"Operation {tid} failed: {json.dumps(res['errors'])}")
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

  async def validate_definitions(self, definitions: list[Definition]):
    data = await self._execute('@ValidateDefinitions', {
      'definitions': definitions,
    })
    return data['validateDefinitions']['warnings']

  async def get_formulation(self, name: str) -> Formulation:
    data = await self._execute('@FetchFormulation', {'name': name})
    form = data['formulation']
    if not form:
      return None
    return Formulation(
      name=form['name'],
      display_name=form['displayName'],
      description=form['description'],
      url=form['url'],
      created_at=form['createdAt'],
    )

  async def update_formulation(
    self,
    name: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    url: Optional[str] = None
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
    await self._execute('@RegisterSpecification', {
      'input': {
        'formulationName': formulation_name,
        'definitions': definitions,
        'tagNames': tag_names,
      }
    })

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
        'absoluteGap': absolute_gap,
        'relativeGap': relative_gap,
        'solveTimeoutMillis': solve_timeout_millis,
        'primalValueEpsilon': primal_value_epsilon,
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
        return _failed_outcome(attempt['outcome']['failure'])
      if status == 'INFEASIBLE':
        return InfeasibleOutcome()
      if status == 'UNBOUNDED':
        return UnboundedOutcome()
      return _feasible_outcome(attempt['outcome'])

  async def get_attempt_template(self, uuid: str) -> AttemptTemplate:
    data = await self._execute('@FetchAttemptTemplate', {'uuid': uuid})
    tpl = data['attempt']['template']
    return AttemptTemplate(
      parameters = [
        Parameter(
          p['label'],
          [ParameterEntry(e['key'], e['value']) for e in p['entries']],
          p['defaultValue']
        )
        for p in tpl['parameters']
      ],
      dimensions = [
        Dimension(d['label'], d['items'])
        for d in tpl['dimensions']
      ]
    )

def _failed_outcome(failure):
  return FailedOutcome(
    status=failure['status'],
    message=failure['message'],
    code=failure.get('code'),
    tags=failure.get('tags')
  )

def _feasible_outcome(outcome):
  return FeasibleOutcome(
    variable_results=[_result(r) for r in outcome['variableResults']],
    is_optimal=outcome['isOptimal'],
    objective_value=outcome['objectiveValue'],
    absolute_gap=outcome.get('absoluteGap')
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
