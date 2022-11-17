import opvious
import os
import pytest


AUTHORIZATION = os.environ.get("OPVIOUS_AUTHORIZATION")


@pytest.fixture
def client():
    return opvious.Client(AUTHORIZATION)


@pytest.mark.skipif(not AUTHORIZATION, reason="No access token detected")
class TestClient:
    @pytest.mark.asyncio
    async def test_run_bounded_feasible_attempt(self, client):
        inputs = await client.assemble_inputs(
            formulation_name="bounded", parameters={"bound": 0.1}
        )
        attempt = await client.start_attempt(inputs)
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)
        assert outcome.is_optimal
        assert outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_run_bounded_infeasible_attempt(self, client):
        inputs = await client.assemble_inputs(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(inputs)
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_simple_unbounded_attempt(self, client):
        inputs = await client.assemble_inputs(formulation_name="unbounded")
        attempt = await client.start_attempt(inputs)
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.UnboundedOutcome)

    @pytest.mark.asyncio
    async def test_run_diet_attempt(self, client):
        inputs = await client.assemble_inputs(
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
            infer_dimensions=True,
        )
        attempt = await client.start_attempt(inputs)
        outcome = await client.wait_for_outcome(attempt)
        assert outcome.is_optimal
        assert outcome.objective_value == 33

        quantities = await client.fetch_variable(attempt, "quantityOfRecipe")
        assert quantities["value"].to_dict() == {("pizza",): 1, ("salad",): 2}

        costs = await client.fetch_parameter(attempt, "costPerRecipe")
        assert costs.to_dict() == {
            ("lasagna",): 12,
            ("pizza",): 15,
            ("salad",): 9,
            ("caviar",): 23,
        }

        nutrients = await client.fetch_constraint(attempt, "enoughNutrients")
        assert nutrients["slack"].to_dict() == {
            ("carbs",): 0,
            ("fibers",): 0,
            ("vitamins",): 1,
        }

    @pytest.mark.asyncio
    async def test_run_relaxed_attempt(self, client):
        inputs = await client.assemble_inputs(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(
            inputs, relaxed_constraints=["greaterThanBound"]
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.FeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_bounded_relaxed_attempt(self, client):
        inputs = await client.assemble_inputs(
            formulation_name="bounded", parameters={"bound": 3}
        )
        attempt = await client.start_attempt(
            inputs,
            relaxed_constraints=[
                opvious.RelaxedConstraint(label="greaterThanBound", bound=0.5)
            ],
        )
        outcome = await client.wait_for_outcome(attempt)
        assert isinstance(outcome, opvious.InfeasibleOutcome)
