import opvious
import os
import pandas as pd
import pytest

ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

@pytest.fixture
def client():
  return opvious.Client(ACCESS_TOKEN)

@pytest.mark.skipif(not ACCESS_TOKEN, reason='No access token detected')
class TestClient:
  async def test_extract_definitions(self, client):
    defs = await client.extract_definitions(
      sources=["""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""]
    )
    assert (len(defs)) == 2

  async def test_compile_valid_spec(self, client):
    defs = await client.extract_definitions(
      sources=["""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""]
    )
    assembly = await client.compile_specification(definitions=defs)
    assert len(assembly['variables']) == 1

  async def test_register_valid_spec(self, client):
    defs = await client.extract_definitions(
      sources=["""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""]
    )
    assembly = await client.register_specification(
      formulation_name='bounded-test',
      definitions=defs
    )
    assert len(assembly['variables']) == 1

  async def test_get_formulation(self, client):
    name = 'get-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^p_p: a \\in \\mathbb{N}
        \\S^v_v: \\alpha \\in \\mathbb{N}
        \\S^c_c: \\alpha \\leq a
        \\S^o: \\max \\alpha
      """
    )
    formulation = await client.get_formulation(name)
    assert formulation.name == name

  async def test_delete_formulation(self, client):
    name = 'delete-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^p_p: a \\in \\mathbb{N}
        \\S^v_v: \\alpha \\in \\mathbb{N}
        \\S^c_c: \\alpha \\leq a
        \\S^o: \\max \\alpha
      """
    )
    await client.delete_formulation(name)
    formulation = await client.get_formulation(name)
    assert formulation is None

  async def test_run_simple_feasible_attempt(self, client):
    name = 'bounded'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^{v}_{power}: \\alpha \\in \\{0,1\\}
        \\S^o: \\max 2 \\alpha
      """
    )
    uuid = await client.start_attempt(formulation_name=name)
    outcome = await client.poll_attempt_outcome(uuid)
    assert outcome.is_optimal
    assert outcome.objective_value == 2

  async def test_run_simple_infeasible_attempt(self, client):
    name = 'infeasible-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^v_a: \\alpha \\in \\{0,1\\}
        \\S^c_n: \\alpha \\leq {-1}
      """
    )
    uuid = await client.start_attempt(formulation_name=name)
    outcome = await client.poll_attempt_outcome(uuid)
    assert isinstance(outcome, opvious.InfeasibleOutcome)

  async def test_run_simple_unbounded_attempt(self, client):
    name = 'unbounded-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^v_a: \\alpha \\in \\mathbb{R}
        \\S^o: \\max \\alpha
      """
    )
    uuid = await client.start_attempt(formulation_name=name)
    outcome = await client.poll_attempt_outcome(uuid)
    assert isinstance(outcome, opvious.UnboundedOutcome)

  async def test_run_diet_attempt(self, client):
    name = 'diet-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source=specification_source('diet')
    )
    cost_per_recipe = {
        'lasagna': 12,
        'pizza': 15,
        'salad': 9,
        'caviar': 23,
    }
    minimal_nutrients = {
        'carbs': 5,
        'vitamins': 3,
        'fibers': 2,
    }
    nutrients_per_recipe = {
        ('carbs', 'lasagna'): 3,
        ('carbs', 'pizza'): 5,
        ('carbs', 'caviar'): 1,
        ('vitamins', 'lasagna'): 1,
        ('vitamins', 'salad'): 2,
        ('vitamins', 'caviar'): 3,
        ('fibers', 'salad'): 1,
    }
    input_dims = [
      opvious.Dimension.iterable('nutrients', minimal_nutrients),
      opvious.Dimension.iterable('recipes', cost_per_recipe),
    ]
    input_params = [
      opvious.Parameter.indexed('costPerRecipe', cost_per_recipe),
      opvious.Parameter.indexed('minimalNutrients', minimal_nutrients),
      opvious.Parameter.indexed('nutrientsPerRecipe', nutrients_per_recipe),
    ]
    uuid = await client.start_attempt(
      formulation_name=name,
      dimensions=input_dims,
      parameters=input_params
    )
    outcome = await client.poll_attempt_outcome(uuid)
    assert outcome.is_optimal
    assert outcome.objective_value == 33
    assert outcome.variable_results == [
      opvious.IndexedResult(
        label='quantityOfRecipe',
        value={'pizza': 1, 'salad': 2},
      )
    ]
    dims = await client.get_attempt_dimensions(uuid)
    assert dims == input_dims
    params = await client.get_attempt_parameters(uuid)
    assert params == input_params

  async def test_compile_diet_specification(self, client):
    defs = await client.extract_definitions([specification_source('diet')])
    assembly = await client.compile_specification(defs)
    assert assembly

  async def test_run_simple_invalid_attempt(self, client):
    name = 'invalid-test'
    await register_specification_from_source(
      client=client,
      formulation_name=name,
      source="""
        \\S^p_p: a \\in \\mathbb{N}
        \\S^v_v: \\alpha \\in \\mathbb{N}
        \\S^c_c: \\alpha \\leq a
        \\S^o: \\max \\alpha
      """
    )
    uuid = await client.start_attempt(
      formulation_name=name,
      parameters=[opvious.Parameter.scalar('p', 0.5)]
    )
    outcome = await client.poll_attempt_outcome(uuid)
    assert isinstance(outcome, opvious.FailedOutcome)
    assert outcome.status == 'INVALID_ARGUMENT'

def specification_source(formulation_name):
  fname = formulation_name + '.md'
  fpath = os.path.join(os.path.dirname(__file__), 'specifications', fname)
  with open(fpath) as reader:
    return reader.read()

async def register_specification_from_source(client, formulation_name, source):
  defs = await client.extract_definitions([source])
  return await client.register_specification(formulation_name, defs)
