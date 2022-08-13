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

_DEFAULT_API_URL = 'https://api.opvious.dev/graphql'

def is_using_pyodide():
  # https://pyodide.org/en/stable/usage/faq.html#how-to-detect-that-code-is-run-with-pyodide
  return 'pyodide' in sys.modules

def pyodide_executor(url, auth):
  from pyodide.http import pyfetch

  class PyodideExecutor:
    """`pyodide`-powered GraphQL executor"""

    async def execute(self, query, variables):
      res = await pyfetch(
        url,
        method='POST',
        headers={'authorization': auth, 'content-type': 'application/json'},
        body=json.dumps({'query': query, 'variables': variables})
      )
      body = await res.js_response.text()
      headers = res.js_response.headers # TODO: This doesn't work.
      return (headers.get('operation'), json.loads(body))

  return PyodideExecutor()

def aiohttp_executor(url, auth):
  import aiohttp

  headers = {'authorization': auth}

  class AiohttpExecutor:
    """`aiohttp`-powered GraphQL executor"""

    async def execute(self, query, variables):
      data = {'query': query, 'variables': variables}
      async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post(url, json=data) as res:
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

  async def compile_specification(self, source_text):
    data = await self._execute('@CompileSpecification', {
      'sourceText': source_text,
    })
    return data['compileSpecification']

  async def update_formulation(
    self,
    name,
    display_name=None,
    description=None
  ):
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

  async def register_specification(
    self,
    formulation_name,
    source_text,
    tags=None
  ):
    data = await self._execute('@RegisterSpecification', {
      'input': {
        'formulationName': formulation_name,
        'sourceText': source_text,
        'tags': tags,
      }
    })
    return data['registerSpecification']

  async def start_attempt(
    self,
    formulation_name,
    specification_tag=None,
    dimensions=None,
    parameters=None,
    relative_gap=None,
    absolute_gap=None,
    primal_value_epsilon=None,
    solve_timeout_millis=None
  ):
    start_data = await self._execute('@StartAttempt', {
      'input': {
        'formulationName': formulation_name,
        'specificationTag': specification_tag,
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

  async def poll_attempt_outcome(self, uuid):
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

  async def get_attempt_parameters(self, uuid):
    data = await self._execute('@FetchAttemptInputs', {'uuid': uuid})
    return [
      Parameter(
        p['label'],
        [ParameterEntry(e['key'], e['value']) for e in p['entries']],
        p['defaultValue']
      )
      for p in data['attempt']['inputs']['parameters']
    ]

def _failed_outcome(data):
  failure = data['failure']
  return FailedOutcome(
    status=failure['status'],
    message=failure['message'],
    code=failure['code'],
    operation=failure['operation'],
    tags=failure['tags']
  )

def _feasible_outcome(outcome, outputs):
  return FeasibleOutcome(
    is_optimal=outcome['isOptimal'],
    objective_value=outcome['objectiveValue'],
    absolute_gap=outcome['absoluteGap'],
    variable_results=[_result(r) for r in outputs['variableResults']],
    constraint_results=[_result(r) for r in outputs['constraintResults']]
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
