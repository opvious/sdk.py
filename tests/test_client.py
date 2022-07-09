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
  async def test_compile_valid_spec(self, client):
    assembly = await client.compile_specification(
      source_text="""\\S^v_a: \\alpha \\in [0,1]; \\S^o: \\max \\alpha"""
    )
    assert len(assembly['variables']) == 1

  async def test_register_valid_spec(self, client):
    spec = await client.register_specification(
      formulation_name='bounded-test',
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

  async def test_run_diet_attempt(self, client):
    name = 'diet-test'
    await client.register_specification(
      formulation_name=name,
      source_text=specification_source('diet')
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
    outcome = await client.run_attempt(
      formulation_name=name,
      dimensions=[
        opvious.Dimension.iterable('nutrients', minimal_nutrients),
        opvious.Dimension.iterable('recipes', cost_per_recipe),
      ],
      parameters=[
        opvious.Parameter.indexed('minimalNutrients', minimal_nutrients),
        opvious.Parameter.indexed('nutrientsPerRecipe', nutrients_per_recipe),
        opvious.Parameter.indexed('costPerRecipe', cost_per_recipe),
      ]
    )
    assert outcome.is_optimal
    assert outcome.objective_value == 33
    assert outcome.variables == [
      opvious.IndexedVariable(
        label='quantityOfRecipe',
        value={'pizza': 1, 'salad': 2},
      )
    ]

  async def test_compile_diet_specification(self, client):
    assembly = await client.compile_specification(specification_source('diet'))
    assert assembly

def specification_source(formulation_name):
  fname = formulation_name + '.md'
  fpath = os.path.join(os.path.dirname(__file__), 'specifications', fname)
  with open(fpath) as reader:
    return reader.read()
