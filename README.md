# Opvious Python SDK  [![CI](https://github.com/opvious/sdk.py/actions/workflows/ci.yml/badge.svg)](https://github.com/mtth/opvious/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

This package provides a lightweight client for interacting with the [Opvious
API][api]. This SDK's functionality is focused on running attempts; for other
operations consider the [TypeScript CLI or SDK][cli].

## Quickstart

First, install this package and have an API access token handy (these can be
generated [here][token]).

```sh
pip install opvious[aio]
```

With these steps out of the way, you are ready to solve any of your optimization
models!

```python
import opvious

# Instantiate an API client from an API token
client = opvious.Client(TOKEN)

# Assemble inputs for a registered formulation
builder = await client.create_inputs_builder('my-formulation')
# Add dimensions and parameters...

# Start an attempt
attempt = await client.start_attempt(builder.build())

# Wait for the attempt to complete
outcome = await attempt.wait_for_outcome()
```

[api]: https://www.opvious.io
[cli]: https://www.opvious.io/sdk.ts
[token]: https://hub.opvious.io/authorizations
