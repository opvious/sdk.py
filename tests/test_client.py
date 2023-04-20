import opvious
import os
import pytest


client = opvious.Client.from_token(
    token=os.environ.get("OPVIOUS_AUTHORIZATION", ""),
    domain=os.environ.get(opvious.Settings.DOMAIN.value),
)


@pytest.mark.skipif(
    not client.authenticated, reason="No access token detected"
)
class TestClient:
    @pytest.mark.asyncio
    async def test_run_bounded_feasible_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="bounded", parameters={"bound": 0.1}
        )
        attempt = await client.start_attempt(request)
        outcome = await client.wait_for_outcome(attempt, assert_feasible=True)
        assert isinstance(outcome, opvious.FeasibleOutcome)
        assert outcome.is_optimal
        assert outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_run_bounded_infeasible_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(request)
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_simple_unbounded_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="unbounded"
        )
        attempt = await client.start_attempt(request)
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.UnboundedOutcome)

    @pytest.mark.asyncio
    async def test_run_diet_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="diet",
            parameters={
                "costPerRecipe": {
                    "lasagna": 12,
                    "pizza": 15,
                    "salad": 9,
                    "caviar": 23,
                },
                "minimalNutrients": {
                    "carbs": 5,
                    "vitamins": 3,
                    "fibers": 2,
                },
                "nutrientsPerRecipe": {
                    ("carbs", "lasagna"): 3,
                    ("carbs", "pizza"): 5,
                    ("carbs", "caviar"): 1,
                    ("vitamins", "lasagna"): 1,
                    ("vitamins", "salad"): 2,
                    ("vitamins", "caviar"): 3,
                    ("fibers", "salad"): 1,
                },
            },
        )
        attempt = await client.start_attempt(request)
        outcome = await client.wait_for_outcome(attempt)
        assert outcome.is_optimal
        assert outcome.objective_value == 33

        input_data = await client.fetch_attempt_inputs(attempt)
        costs = input_data.parameter("costPerRecipe")
        assert costs.to_dict() == {
            "lasagna": 12,
            "pizza": 15,
            "salad": 9,
            "caviar": 23,
        }

        output_data = await client.fetch_attempt_outputs(attempt)
        quantities = output_data.variable("quantityOfRecipe")
        assert quantities["value"].to_dict() == {"pizza": 1, "salad": 2}
        nutrients = output_data.constraint("enoughNutrients")
        assert nutrients["slack"].to_dict() == {
            "carbs": 0,
            "fibers": 0,
            "vitamins": 1,
        }

    @pytest.mark.asyncio
    async def test_run_pinned_diet_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="diet",
            parameters={
                "costPerRecipe": {
                    "lasagna": 10,
                    "pizza": 20,
                },
                "minimalNutrients": {
                    "carbs": 5,
                },
                "nutrientsPerRecipe": {
                    ("carbs", "lasagna"): 1,
                    ("carbs", "pizza"): 1,
                },
            },
        )
        attempt = await client.start_attempt(
            request=request,
            pinned_variables={
                "quantityOfRecipe": {"pizza": 1},
            },
        )
        outcome = await client.wait_for_outcome(attempt)
        assert outcome.is_optimal
        assert outcome.objective_value == 60

        output_data = await client.fetch_attempt_outputs(attempt)
        quantities = output_data.variable("quantityOfRecipe")
        assert quantities["value"].to_dict() == {"pizza": 1, "lasagna": 4}

    @pytest.mark.asyncio
    async def test_run_relaxed_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(
            request,
            relaxation=opvious.Relaxation.from_constraint_labels(
                [
                    "greaterThanBound",
                ]
            ),
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_bounded_relaxed_attempt(self):
        request = await client.prepare_attempt_request(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(
            request,
            relaxation=opvious.Relaxation(
                penalty="MAX_DEVIATION",
                objective_weight=1,
                constraints=[
                    opvious.ConstraintRelaxation(
                        label="greaterThanBound",
                        bound=1,
                    ),
                ],
            ),
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_sudoku(self):
        request = await client.prepare_attempt_request(
            formulation_name="sudoku",
            parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
        )
        attempt = await client.start_attempt(request)

        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)

        output_data = await client.fetch_attempt_outputs(attempt)
        decisions = output_data.variable("decisions")
        assert (0, 0, 3) in decisions.index

    @pytest.mark.asyncio
    async def test_solve_bounded_feasible(self):
        outputs = await client.run_solve(
            sources=[
                r"""
                    $\S^{v}_{target}: \alpha \in \{0,1\}$
                    $\S^{p}_{bound}: b \in \mathbb{R}_+$
                    $\S^{c}_{greaterThanBound}: \alpha \geq b$
                    $\S^o_{maximize}: \max 2 \alpha$
                """,
            ],
            parameters={"bound": 0.1},
        )
        assert isinstance(outputs.outcome, opvious.FeasibleOutcome)
        assert outputs.outcome.is_optimal
        assert outputs.outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_solve_bounded_infeasible(self):
        outputs = await client.run_solve(
            sources=[
                r"""
                    $\S^{v}_{target}: \alpha \in \{0,1\}$
                    $\S^{p}_{bound}: b \in \mathbb{R}_+$
                    $\S^{c}_{greaterThanBound}: \alpha \geq b$
                    $\S^o_{maximize}: \max 2 \alpha$
                """,
            ],
            parameters={"bound": 30},
        )
        assert isinstance(outputs.outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_portfolio_selection(self):
        source = r"""
            We find an allocation of assets which minimizes risk while
            satisfying a minimum expected return and allocation per group.

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
        outputs = await client.run_solve(
            sources=[source],
            parameters={
                "covariance": {
                    ("AAPL", "AAPL"): 0.2,
                    ("AAPL", "MSFT"): 0.1,
                    ("MSFT", "AAPL"): 0.1,
                    ("MSFT", "MSFT"): 0.25,
                },
                "expectedReturn": {
                    "AAPL": 0.15,
                    "MSFT": 0.2,
                },
                "desiredReturn": 0.1,
            },
        )
        assert isinstance(outputs.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_diet(self):
        outputs = await client.run_solve(
            formulation_name="diet",
            parameters={
                "costPerRecipe": {
                    "lasagna": 12,
                    "pizza": 15,
                },
                "minimalNutrients": {
                    "carbs": 5,
                },
                "nutrientsPerRecipe": {
                    ("carbs", "lasagna"): 3,
                    ("carbs", "pizza"): 5,
                },
            },
            options=opvious.SolveOptions(free_bound_threshold=1e9),
        )
        assert isinstance(outputs.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_no_objective(self):
        source = r"""
        # N queens

        First, let $\S^p_{size}: n \in \mathbb{N}$ be the size of the board.
        Given this, we define $\S^a: N \doteq \{1 \ldots n\}$ the set of
        possible positions and $\S^v_{decisions}: \alpha \in \{0,1\}^{N \times
        N}$ our optimization variable. A valid board must satisfy the following
        constraints:

        $$
        \begin{align}
            \S^c_{onePerRow}:
                \forall i \in N, \sum_{j \in N} \alpha_{i,j} = 1 \\
            \S^c_{onePerColumn}:
                \forall j \in N, \sum_{i \in N} \alpha_{i,j} = 1 \\
            \S^c_{onePerDiag1}:
                \forall d \in \{2 \ldots 2 n\},
                \sum_{i \in N \mid d - i \in N} \alpha_{i,d-i} \leq 1 \\
            \S^c_{onePerDiag2}:
                \forall d \in \{1-n \ldots n-1\},
                \sum_{i \in N \mid i - d \in N} \alpha_{i,i-d} \leq 1 \\
        \end{align}
        $$
        """
        outputs = await client.run_solve(
            sources=[source],
            parameters={"size": 2},
        )
        assert isinstance(outputs.outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_inspect(self):
        instructions = await client.inspect_solve_instructions(
            formulation_name="sudoku",
            parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
        )
        assert "decisions" in instructions
