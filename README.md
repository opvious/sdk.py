# Opvious Python SDK  [![CI](https://github.com/opvious/sdk.py/actions/workflows/ci.yml/badge.svg)](https://github.com/opvious/sdk.py/actions/workflows/ci.yml) [![Pypi badge](https://badge.fury.io/py/opvious.svg)](https://pypi.python.org/pypi/opvious/)

An optimization SDK for solving linear, mixed-integer, and quadratic models

## Highlights

### Declarative modeling API

+ Extensive static validations
+ Exportable to LaTeX
+ Extensible support for high-level patterns (activation variables, masks, ...)

```python
import opvious.modeling as om

class BinPacking(om.Model):
  items = om.Dimension()
  weight = om.Parameter.non_negative(items)
  bins = om.interval(1, om.size(items))
  bin_max_weight = om.Parameter.non_negative()

  assigned = om.Variable.indicator(items, bins)
  bin_used = om.Variable.indicator(bins)

  @om.objective
  def minimize_bins_used(self):
    return om.total(self.bin_used(b) for b in self.bins)

  @om.constraint
  def each_item_is_assigned_once(self):
    for i in self.items:
      yield om.total(self.assigned(i, b) for b in self.bins) == 1

  @om.constraint
  def bins_with_assignments_are_used(self):
    for i, b in om.cross(self.items, self.bins):
      yield self.assigned(i, b) <= self.bin_used(b)

  @om.constraint
  def bin_weights_are_below_max(self):
    for b in self.bins:
      bin_weight = om.total(self.weight(i) * self.assigned(i, b) for i in self.items)
      yield bin_weight <= self.bin_max_weight()
```


### Transparent remote solves

+ No local solver installation required
+ Real-time progress notifications
+ Seamless data import/export via native support for `pandas`
+ Flexible multi-objective support: weighted sums, epsilon constraints, ...
+ Built-in debugging capabilities: relaxations, fully annotated LP formatting,
  ...

```python
import opvious

client = opvious.Client.from_environment()

response = await client.run_solve(
  specification=BinPacking().specification(),
  parameters={
    "weight": {"a": 10.5, "b": 22, "c": 48},
    "binMaxWeight": 50,
  },
)
solution = response.outputs.variable("assigned")  # Optimal assignment dataframe
```

Take a look at https://opvious.readthedocs.io for the full documentation or
[these notebooks][notebooks] to see the SDK in action.

[notebooks]: https://github.com/opvious/examples/tree/main/notebooks
