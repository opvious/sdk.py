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
        builder = await client.create_inputs_builder("bounded")
        builder.set("bound", 0.1)
        attempt = await client.start_attempt(builder.build())
        outcome = await attempt.wait_for_outcome()
        assert outcome.is_optimal
        assert outcome.objective_value == 2

    @pytest.mark.asyncio
    async def test_run_bounded_infeasible_attempt(self, client):
        builder = await client.create_inputs_builder("bounded")
        builder.set("bound", 3)
        attempt = await client.start_attempt(builder.build())
        outcome = await attempt.wait_for_outcome()
        assert isinstance(outcome, opvious.InfeasibleOutcome)

    @pytest.mark.asyncio
    async def test_run_simple_unbounded_attempt(self, client):
        builder = await client.create_inputs_builder("unbounded")
        attempt = await client.start_attempt(builder.build())
        outcome = await attempt.wait_for_outcome()
        assert isinstance(outcome, opvious.UnboundedOutcome)

    @pytest.mark.asyncio
    async def test_run_diet_attempt(self, client):
        builder = await client.create_inputs_builder("diet")
        builder.set(
            "costPerRecipe",
            {
                "lasagna": 12,
                "pizza": 15,
                "salad": 9,
                "caviar": 23,
            },
        )
        builder.set(
            "minimalNutrients",
            {
                "carbs": 5,
                "vitamins": 3,
                "fibers": 2,
            },
        )
        builder.set(
            "nutrientsPerRecipe",
            {
                ("carbs", "lasagna"): 3,
                ("carbs", "pizza"): 5,
                ("carbs", "caviar"): 1,
                ("vitamins", "lasagna"): 1,
                ("vitamins", "salad"): 2,
                ("vitamins", "caviar"): 3,
                ("fibers", "salad"): 1,
            },
        )
        attempt = await client.start_attempt(builder.build(True))
        outcome = await attempt.wait_for_outcome()
        assert outcome.is_optimal
        assert outcome.objective_value == 33

        quantities = await attempt.load_variable_result("quantityOfRecipe")
        assert quantities["value"].to_dict() == {("pizza",): 1, ("salad",): 2}

        costs = await attempt.load_parameter("costPerRecipe")
        assert costs.to_dict() == {
            ("lasagna",): 12,
            ("pizza",): 15,
            ("salad",): 9,
            ("caviar",): 23,
        }

        nutrients = await attempt.load_constraint_result("enoughNutrients")
        assert nutrients["slack"].to_dict() == {
            ("carbs",): 0,
            ("fibers",): 0,
            ("vitamins",): 1,
        }
