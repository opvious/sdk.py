# Opvious Python SDK  [![CI](https://github.com/opvious/sdk.py/actions/workflows/ci.yml/badge.svg)](https://github.com/opvious/sdk.py/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

This package provides a lightweight SDK for solving optimization models with the
[Opvious API][api]. Its main features are:

+ Seamless data import/export via native support for [`pandas`][pandas]
+ Powerful built-in debugging capabilities: automatic infeasibility relaxation,
  variable pinning, and more
+ Non-blocking APIs for performant parallel calls


## Quickstart

First, install this package and have an API access token handy (these can be
generated [here][token]).

```sh
pip install opvious[aio] # aio is recommended for improved performance
```

With these steps out of the way, you are ready to optimize!

```python
import opvious

client = opvious.Client.from_token(TOKEN)

# Solve a simple portfolio selection optimization model
response = await client.run_solve(
    specification=opvious.InlineSpecification([
      r"""
          We find an allocation of assets which minimizes risk while satisfying
          a minimum expected return:

          + A collection of assets: $\S^d_{asset}: A$
          + Covariances: $\S^p_{covariance}: c \in \mathbb{R}^{A \times A}$
          + Expected return: $\S^p_{expectedReturn}: m \in \mathbb{R}^A$
          + Minimum desired return: $\S^p_{desiredReturn}: r \in \mathbb{R}$

          The only output is the allocation per asset
          $\S^v_{allocation}: \alpha \in [0,1]^A$ chosen to minimize risk:
          $\S^o_{risk}: \min \sum_{a, b \in A} c_{a,b} \alpha_a \alpha_b$.

          Subject to the following constraints:

          + $\S^c_{atLeastMinimumReturn}: \sum_{a \in A} m_a \alpha_a \geq r$
          + $\S^c_{totalAllocation}: \sum_{a \in A} \alpha_a = 1$
      """
    ]),
    parameters={
        "covariance": {
            ("AAPL", "AAPL"): 0.08,
            # ...
        },
        "expectedReturn": {
            "AAPL": 0.07,
            # ..
        },
        "desiredReturn": 0.05,
    },
)

print(f"Problem was {response.status}.") # OPTIMAL, INFEASIBLE, ...
if response.outputs: # Present if the problem was feasible
  print(response.outputs.variable("allocation"))
```


## Environments

Clients are compatible with Pyodide environments, for example [JupyterLite][]
kernels. Simply install the package as usual in a notebook, omitting the `aio`
optional dependencies:

```python
import piplite
await piplite.install('opvious')
```

In other environments, prefer using the `aiohttp`-powered clients as they are
more performant (this is the default if the `aio` dependencies were specified).


## Next steps

This SDK is focused on solving optimization models. For convenient access to the
rest of Opvious' functionality, consider using the [TypeScript SDK and
CLI][cli].


[api]: https://www.opvious.io
[cli]: https://www.opvious.io/sdk.ts
[JupyterLite]: https://jupyterlite.readthedocs.io/
[token]: https://hub.beta.opvious.io/authorizations
[pandas]: https://pandas.pydata.org
