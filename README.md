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

```py
import opvious

client = opvious.Client(ACCESS_TOKEN)

# Attempt to solve a formulation
attempt = await client.start_attempt(
  formulation_name='my-model',
  # inputs...
)

# Wait for the attempt to complete
outcome = await client.poll_attempt_outcome(uuid)
```

[Opvious API]: https://www.opvious.io/
[Typescript SDK]: https://www.opvious.io/sdk.ts/
