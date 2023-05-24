.. default-role:: code

Multi-objective strategies
==========================

Specifications can include zero, one, or many objectives. When there is none or 
just one, no explicit strategy is needed:

* Without an objective, the model checks for feasibility
* With a single objective, the model optimizes this objective

When more than one objective is present, a :class:`.SolveStrategy` is needed to 
define how to pick the optimization target. It exposes two main building blocks, 
described in this page.

.. contents:: Table of contents
   :local:
   :backlinks: none


Weighted sum targets
********************

Strategy targets allow optimizing the weighted sum of multiple objectives by 
passing in a mapping from objective label to weight:

.. code-block:: python

  # Minimize the weighted sum of two objectives
  strategy = opvious.SolveStrategy(
    target={"minimizeFoo": 2, "minimizeBar": 10}
  )

This is possible even if the objectives have difference senses, in that case we 
just need to specify which sense to pick for the final objective:

.. code-block:: python

  # Minimize the weighted sum of two objectives
  strategy = opvious.SolveStrategy(
    sense="MINIMIZE",
    target={"minimizeFoo": 2, "maximizeBar": 10},
  )

All objectives must have an associated non-negative weight in the mapping 
(possibly 0 to ignore the objective). You can use a `defaultdict`_ to set a 
default weight:

.. code-block:: python

  import collections

  # Minimize the weighted sum of all objectives where `minimizeFoo` has weight 2
  # and all others weight 1.
  strategy = opvious.SolveStrategy(
    sense="MINIMIZE",
    target=collections.defaultdict(lambda: 1, {"minimizeFoo": 2}),
  )

As a convenience:

+ Passing in a label string instead of a mapping is equivalent to optimizing 
  only that objective.
+ Calling :meth:`.SolveStrategy.equally_weighted_sum` is equivalent to a
  mapping where all objectives have equal weight 1.


Epsilon constraints
*******************

A complementary approach to optimizing a linear combination of objectives once 
is to iteratively optimize a sequence of objectives, progressively restricting 
the model to be within a given small tolerance ("epsilon") of each prior optimal 
solution.

.. code-block:: python

  # Minimize foo while ensuring that bar is within 10% of its optimal value
  strategy = opvious.SolveStrategy(
    target="minimizeFoo",
    epsilon_constraints=[
      opvious.EpsilonConstraint(target="maximizeBar", relative_tolerance=0.1),
    ],
  )

It's possible to specify many epsilon constraints, they will be applied in the 
order they are defined. Epsilon targets can be single objectives or weighted 
sums but should not include quadratic costs.

.. note::
   Under the hood, the solver incrementally adds epsilon constraints on top of 
   the original model and automatically warm starts from the previous solution. 
   This approach drastically reduces the overhead of each epsilon-constraint.


.. _defaultdict: https://docs.python.org/3/library/collections.html#defaultdict-objects
