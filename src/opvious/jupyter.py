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

from IPython.display import Javascript
import json

from .client import REGISTER_SPECIFICATION_QUERY

_FORMULATION_URL_PREFIX = 'https://console.opvious.dev/formulations/'

def save_specification(client, formulation_name):
  """
  Aggregates all MathJax rendered equations in the currently active notebook and
  registers the combined content as a specification.
  """
  # This implementation is a (brittle...) hack to work around the inability to
  # access notebook cells from Python.
  src = f"""
    (async () => {{
      const notebookId = document
        .querySelector('li[data-type="document-title"][aria-selected="true"]')
        .getAttribute('data-id');
      const renderedCells = document
        .getElementById(notebookId)
        .querySelectorAll('script[type^="math/tex"]');
      const source = [...renderedCells].map((e) => e.textContent).join(' ');
      const res = await fetch(
        {json.dumps(client.api_url)},
        {{
          method: 'POST',
          headers: {{
            authorization: {json.dumps(client.authorization_header)},
            'content-type': 'application/json',
          }},
          body: JSON.stringify({{
            query: `{REGISTER_SPECIFICATION_QUERY}`,
            variables: {{
              input: {{
                formulationName: {json.dumps(formulation_name)},
                sourceText: source.replace(/\s+/g, ' ')
              }}
            }}
          }}),
        }}
      );
      const body = await res.json();
      if (body.errors) {{
        if (
          body.errors.length !== 1 ||
          body.errors[0].extensions?.status !== 'INVALID_ARGUMENT' ||
          !body.errors[0].extensions.exception?.tags?.reason
        ) {{
          reportFailure(body.errors);
          return;
        }}
        element.innerHTML = `
          This specification is invalid, please fix the following error before
          attempting to register it again:
          <p style="color: red; margin-top:2px;">
            ${{body.errors[0].extensions.exception.tags.reason}}
          </p>
        `;
        return;
      }}
      const {{formulation}} = body.data.registerSpecification;
      const surl = {json.dumps(_FORMULATION_URL_PREFIX)} + formulation.name;
      element.innerHTML = `
        Specification successfully created:
        <a href="${{surl}}" target="_blank" style="text-decoration: underline;">
          ${{surl}}
        </a>
      `;
    }})().catch(reportFailure);

    function reportFailure(arg) {{
      element.innerHTML = `
        An unexpected error occurred. Please see console for more information.
      `;
      console.error(arg);
    }}
  """
  return Javascript(src)
