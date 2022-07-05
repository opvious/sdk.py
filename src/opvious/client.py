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

from .data import *

_DEFAULT_API_URL = 'https://api.opvious.dev/graphql'

_COMPILE_SPECIFICATION_QUERY = """
  query CompileSpecification($sourceText: String!) {
    compileSpecification(sourceText: $sourceText)
  }
"""

REGISTER_SPECIFICATION_QUERY = """
  mutation RegisterSpecification($input: RegisterSpecificationInput!) {
    registerSpecification(input: $input) {
      id
      formulation {
        name
      }
      assembly
    }
  }
"""

_RUN_ATTEMPT_QUERY = """
  mutation RunAttempt($input:AttemptInput!) {
    runAttempt(input:$input) {
      outcome {
        __typename
        ...on FeasibleOutcome {
          isOptimal
          objectiveValue
          relativeGap
          variables {
            label
            results {
              key
              primalValue
            }
          }
        }
      }
    }
  }
"""

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
      return json.loads(body)

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
          return await res.json()

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

  async def compile_specification(self, source_text):
    res = await self._executor.execute(_COMPILE_SPECIFICATION_QUERY, {
      'sourceText': source_text,
    })
    if res.get('errors'):
      raise Exception(json.dumps(res['errors']))
    return res['data']['compileSpecification']

  async def register_specification(self, formulation_name, source_text):
    res = await self._executor.execute(REGISTER_SPECIFICATION_QUERY, {
      'input': {
        'formulationName': formulation_name,
        'sourceText': source_text,
      }
    })
    if res.get('errors'):
      raise Exception(json.dumps(res['errors']))
    return res['data']['registerSpecification']

  async def run_attempt(
    self,
    formulation_name,
    collections=None,
    parameters=None,
    relative_gap=None,
    primal_value_epsilon=None,
    solve_timeout_millis=None,
  ):
    res = await self._executor.execute(_RUN_ATTEMPT_QUERY, {
      'input': {
        'formulationName': formulation_name,
        'collections': [c.to_input() for c in collections] if collections else [],
        'parameters': [p.to_input() for p in parameters] if parameters else [],
        'options': {
          'relativeGap': relative_gap,
          'solveTimeoutMillis': solve_timeout_millis,
          'primalValueEpsilon': primal_value_epsilon,
        },
      }
    })
    if res.get('errors'):
      raise Exception(json.dumps(res['errors']))
    data = res['data']['runAttempt']['outcome']
    typename = data['__typename']
    if typename == 'FailedOutcome':
      return _failed_outcome(data)
    if typename == 'FeasibleOutcome':
      return _feasible_outcome(data)
    if typename == 'InfeasibleOutcome':
      return InfeasibleOutcome()
    if typename == 'UnboundedOutcome':
      return UnboundedOutcome()

def _failed_outcome(data):
  return FailedOutcome(error_messages=[e['message'] for e in data['errors']])

def _feasible_outcome(data):
  return FeasibleOutcome(
    is_optimal=data['isOptimal'],
    objective_value=data['objectiveValue'],
    relative_gap=data['relativeGap'],
    variables=[_variable(v) for v in data['variables']]
  )

def _variable(data):
  label = data['label']
  results = data['results']
  if len(results) == 1 and not results[0]['key']:
    return ScalarVariable(label=label, value=results[0]['primalValue'])
  index = [tuple(r['key']) for r in results]
  data = [r['primalValue'] for r in results]
  return IndexedVariable(label=label, value=pd.Series(data=data, index=index))
