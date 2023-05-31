import opvious
import pytest


client = opvious.Client.from_environment()


@pytest.mark.skipif(
    not client.authenticated, reason="No access token detected"
)
class TestClient:
    @pytest.mark.asyncio
    async def test_run_bounded_feasible_attempt(self):
        attempt = await client.start_attempt(
            specification=opvious.FormulationSpecification("bounded"),
            parameters={"bound": 0.1},
        )
        outcome = await client.wait_for_outcome(attempt, assert_feasible=True)
        assert isinstance(outcome, opvious.FeasibleOutcome)
        assert outcome.is_optimal
        assert outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_run_bounded_infeasible_attempt(self):
        attempt = await client.start_attempt(
            specification="bounded", parameters={"bound": 3}
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_simple_unbounded_attempt(self):
        attempt = await client.start_attempt(specification="unbounded")
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.UnboundedOutcome)

    @pytest.mark.asyncio
    async def test_run_diet_attempt(self):
        attempt = await client.start_attempt(
            specification="diet",
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
    async def test_run_relaxed_attempt(self):
        attempt = await client.start_attempt(
            specification="bounded",
            transformations=[
                opvious.transformations.RelaxConstraints(["greaterThanBound"]),
            ],
            strategy=opvious.SolveStrategy(
                target="greaterThanBound_minimizeDeficit"
            ),
            parameters={"bound": 3},
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)
        assert outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_run_bounded_relaxed_attempt(self):
        attempt = await client.start_attempt(
            specification="bounded",
            transformations=[
                opvious.transformations.RelaxConstraints(
                    labels=["greaterThanBound"],
                    penalty="MAX_DEVIATION",
                    is_capped=True,
                ),
            ],
            strategy=opvious.SolveStrategy.equally_weighted_sum("MINIMIZE"),
            parameters={
                "bound": 3,
                "greaterThanBound_deficitCap": 1,
            },
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_sudoku(self):
        attempt = await client.start_attempt(
            specification="sudoku",
            parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
        )

        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)

        output_data = await client.fetch_attempt_outputs(attempt)
        decisions = output_data.variable("decisions")
        assert (0, 0, 3) in decisions.index

    @pytest.mark.asyncio
    async def test_solve_bounded_feasible(self):
        spec = opvious.LocalSpecification.inline(
            r"""
            $\S^{v}_{target}: \alpha \in \{0,1\}$
            $\S^{p}_{bound}: b \in \mathbb{R}_+$
            $\S^{c}_{greaterThanBound}: \alpha \geq b$
            $\S^o_{maximize}: \max 2 \alpha$
            """
        )
        res = await client.run_solve(
            specification=spec,
            parameters={"bound": 0.1},
        )
        assert isinstance(res.outcome, opvious.FeasibleOutcome)
        assert res.outcome.is_optimal
        assert res.outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_solve_bounded_infeasible(self):
        spec = opvious.LocalSpecification.inline(
            r"""
            $\S^{v}_{target}: \alpha \in \{0,1\}$
            $\S^{p}_{bound}: b \in \mathbb{R}_+$
            $\S^{c}_{greaterThanBound}: \alpha \geq b$
            $\S^o_{maximize}: \max 2 \alpha$
            """
        )
        res = await client.run_solve(
            specification=spec,
            parameters={"bound": 30},
        )
        assert isinstance(res.outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_portfolio_selection(self):
        spec = opvious.LocalSpecification.inline(
            r"""
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
        )
        response = await client.run_solve(
            specification=spec,
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
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_diet(self):
        response = await client.run_solve(
            specification=opvious.FormulationSpecification("diet"),
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
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_diet_non_streaming(self):
        response = await client.run_solve(
            specification=opvious.FormulationSpecification("diet"),
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
            prefer_streaming=False,
        )
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_assert_feasible(self):
        try:
            await client.run_solve(
                specification=opvious.FormulationSpecification("diet"),
                parameters={
                    "costPerRecipe": {
                        "pasta": 10,
                    },
                    "minimalNutrients": {
                        "carbs": 5,
                    },
                    "nutrientsPerRecipe": {},  # No carbs
                },
                assert_feasible=True,
            )
            raise Exception()
        except opvious.UnexpectedOutcomeError as exc:
            assert isinstance(exc.outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_no_objective(self):
        spec = opvious.LocalSpecification.inline(
            r"""
            # N queens

            First, let $\S^p_{size}: n \in \mathbb{N}$ be the size of the
            board. Given this, we define $\S^a: N \doteq \{1 \ldots n\}$ the
            set of possible positions and $\S^v_{decisions}: \alpha \in
            \{0,1\}^{N \times N}$ our optimization variable. A valid board must
            satisfy the following constraints:

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
        )
        response = await client.run_solve(
            specification=spec,
            parameters={"size": 2},
        )
        assert isinstance(response.outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_solve_relaxed_sudoku(self):
        response = await client.run_solve(
            specification=opvious.FormulationSpecification("sudoku"),
            parameters={"hints": [(0, 0, 3), (1, 1, 3)]},
            transformations=[
                opvious.transformations.RelaxConstraints(["hintsObserved"]),
            ],
        )
        assert isinstance(response.outcome, opvious.FeasibleOutcome)
        deficit = response.outputs.variable("hintsObserved_deficit")
        assert len(deficit) == 1

    @pytest.mark.asyncio
    async def test_inspect_solve_instructions(self):
        instructions = await client.inspect_solve_instructions(
            specification=opvious.FormulationSpecification("sudoku"),
            parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
        )
        assert "decisions" in instructions

    @pytest.mark.asyncio
    async def test_solve_sudoku_from_url(self):
        response = await client.run_solve(
            specification=opvious.RemoteSpecification.example("sudoku"),
            parameters={"input": [(0, 0, 3)]},
        )
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_weighted_sum_objective(self):
        response = await client.run_solve(
            specification=opvious.FormulationSpecification("group-expenses"),
            parameters={
                "paid": {
                    ("t1", "ann"): 10,
                    ("t2", "ann"): 10,
                    ("t2", "bob"): 10,
                },
                "share": {
                    ("t1", "ann"): 1,
                    ("t1", "bob"): 1,
                    ("t2", "bob"): 1,
                },
                "floor": 0,
            },
            strategy=opvious.SolveStrategy(
                target="minimizeIndividualTransfers",
                epsilon_constraints=[
                    opvious.EpsilonConstraint(
                        target="minimizeTotalTransferred",
                        relative_tolerance=0.1,
                    ),
                ],
            ),
        )
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_relaxes_all_constraints(self):
        response = await client.run_solve(
            specification=opvious.FormulationSpecification("group-expenses"),
            parameters={
                "paid": {
                    ("t1", "ann"): 10,
                },
                "share": {
                    ("t1", "ann"): 1,
                },
                "floor": 10,
            },
            transformations=[
                opvious.transformations.RelaxConstraints(),
            ],
            strategy=opvious.SolveStrategy.equally_weighted_sum(),
        )
        assert isinstance(response.outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_save_specification(self):
        ns = opvious.load_notebook_models(
            "notebooks/set-cover.ipynb", root=__file__
        )
        spec = await client.save_specification(
            ns.model.specification(), formulation_name="set-cover-notebook"
        )
        assert isinstance(spec, opvious.FormulationSpecification)
        res = await client.run_solve(
            specification=spec,
            parameters={"covers": [("s", "v")]},
        )
        assert isinstance(res.outcome, opvious.FeasibleOutcome)
