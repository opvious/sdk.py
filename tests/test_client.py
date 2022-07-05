import opvious
import os
import pytest

API_TOKEN = os.environ.get('API_TOKEN')

@pytest.fixture
def client():
  return opvious.Client(API_TOKEN)

@pytest.mark.skipif(not API_TOKEN, reason='No API token detected')
class TestClient:
  async def test_compile_valid_spec(self, client):
    assembly = await client.compile_specification(
      source_text="""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""
    )
    assert len(assembly['variables']) == 1

  async def test_register_valid_spec(self, client):
    spec = await client.register_specification(
      formulation_name='bounded',
      source_text="""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""
    )
    assert spec.get('assembly')

  async def test_run_simple_feasible_attempt(self, client):
    name = 'bounded'
    await client.register_specification(
      formulation_name=name,
      source_text="""
        \\S^{v}_{power}: \\alpha \\in \\{0,1\\}
        \\S^o: \\max 2 \\alpha
      """
    )
    outcome = await client.run_attempt(formulation_name=name)
    assert outcome.is_optimal
    assert outcome.objective_value == 2

  async def test_run_simple_infeasible_attempt(self, client):
    name = 'infeasible-test'
    await client.register_specification(
      formulation_name=name,
      source_text="""
        \\S^v_a: \\alpha \\in \\{0,1\\}
        \\S^c_n: \\alpha \\leq {-1}
      """
    )
    outcome = await client.run_attempt(formulation_name=name)
    assert isinstance(outcome, opvious.InfeasibleOutcome)

  async def test_run_simple_unbounded_attempt(self, client):
    name = 'unbounded-test'
    await client.register_specification(
      formulation_name=name,
      source_text="""
        \\S^v_a: \\alpha \\in \\mathbb{R}
        \\S^o: \\max \\alpha
      """
    )
    outcome = await client.run_attempt(formulation_name=name)
    assert isinstance(outcome, opvious.UnboundedOutcome)
