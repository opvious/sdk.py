# Opvious Python SDK  [![CI](https://github.com/opvious/sdk.py/actions/workflows/ci.yml/badge.svg)](https://github.com/opvious/sdk.py/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

This package provides a lightweight SDK for solving optimization models with the
[Opvious API][api]. Its main features are:

+ Seamless data import/export via native support for [`pandas`][pandas]
+ Powerful built-in debugging capabilities: automatic infeasibility relaxation,
  variable pinning, and more
+ Non-blocking APIs for performant parallel optimization

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

# Assemble and validate inputs for a registered formulation
inputs = await client.assemble_inputs(
    formulation_name='my-formulation',
    parameters={
        # Formulation parameters...
    }
)

# Start an attempt and wait for it to complete
attempt = await client.start_attempt(inputs)

# Wait for the attempt to complete
outcome = await client.wait_for_outcome(attempt)
```

[api]: https://www.opvious.io
[cli]: https://www.opvious.io/sdk.ts
[token]: https://hub.opvious.io/authorizations
[pandas]: https://pandas.pydata.org

## Next steps

This SDK is focused on solving optimization models. For convenient access to the
rest of Opvious API's functionality, consider using the [TypeScript SDK and
CLI][cli].
