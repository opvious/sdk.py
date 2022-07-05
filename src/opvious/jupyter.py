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
  Aggregates all Markdown cells in the notebook and registers the combined
  content as a specification.
  """
  # This implementation is a big hack to work around the inability to access
  # notebook cells from Python.
  # TODO: Improve output (particularly any errors...)
  src = f"""
    (async () => {{
      const source = [...document.getElementsByClassName('jp-MarkdownCell')]
        .map(
          (e) => [...e.querySelectorAll('.CodeMirror-line')]
            .map((l) => l.textContent)
            .join(' ')
          )
        .join(' ');
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
        element.innerHTML = `
          <pre>${{JSON.stringify(body.errors, null, 2)}}</pre>
        `;
        return;
      }}
      const spec = body.data.registerSpecification;
      const specUrl = {json.dumps(_FORMULATION_URL_PREFIX)} + spec.formulation.name;
      element.innerHTML = `
        Specification successfully created:
        <a href="${{specUrl}}" target="_blank">${{specUrl}}</a>
      `;
    }})().catch(console.error);
  """
  return Javascript(src)
