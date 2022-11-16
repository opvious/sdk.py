# Opvious Python SDK  [![CI](https://github.com/mtth/opvious/actions/workflows/ci.yml/badge.svg)](https://github.com/mtth/opvious/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

This package provides a lightweight client for interacting with the [Opvious
API][]. This SDK's functionality is focused on running attempts; for other
operations consider the [TypeScript CLI or SDK][].

## Quickstart

First, to install this package:

```sh
pip install opvious[aio]
```

You'll then need an API access token. You can generate one at
https://hub.opvious.io/authorizations. Once you have it, you can
instantiate a client and call its method:

```python
import opvious

# Instantiate an API client
client = opvious.Client(TOKEN)

# Assemble problem inputs
builder = await client.create_inputs_builder('my-formulation')
# Add dimensions and parameters...

# Start an attempt
attempt = await client.start_attempt(builder.build())

# Wait for the attempt to complete
outcome = await attempt.wait_for_outcome()
```

[Opvious API]: https://www.opvious.io/
[Typescript SDK]: https://www.opvious.io/sdk.ts/
