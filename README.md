# Opvious Python SDK  [![CI](https://github.com/opvious/sdk.py/actions/workflows/ci.yml/badge.svg)](https://github.com/opvious/sdk.py/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

An optimization SDK for solving linear, mixed-integer, and quadratic models

```python
import opvious

client = opvious.Client.from_environment()

# Solve a portfolio selection optimization model
response = await client.run_solve(
    specification=opvious.LocalSpecification.inline(
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
    ),
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
    assert_feasible=True,
)

optimal_allocation = response.outputs.variable("allocation")
```

Take a look at https://opvious.readthedocs.io for the full documentation or
[these notebooks][notebooks] to see it in action.

[notebooks]: https://github.com/opvious/examples/tree/main/notebooks
