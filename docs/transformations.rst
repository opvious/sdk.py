.. default-role:: code

Model transformations
=====================

Transformations allow changing the solved model without updating its 
specification. Common use-cases include:

+ Relaxing a constraint to investigate infeasibilities
+ Pinning a variable to estimate the impact of an override
+ Omitting a constraint to see its impact on the objective value

.. contents:: Table of contents
   :local:
   :backlinks: none


Base transformations
********************

This section lists available transformation building blocks. These can be used 
individually or combined (see :ref:`Transformation patterns`).

Pinning variables
-----------------

.. autoclass:: opvious.PinVariables
   :noindex:
   :members:
   :show-inheritance:


Relaxing constraints
--------------------

.. autoclass:: opvious.RelaxConstraints
   :noindex:
   :members:
   :show-inheritance:


Densifying variables
--------------------

.. autoclass:: opvious.DensifyVariables
   :noindex:
   :members:
   :show-inheritance:


Omitting constraints and objectives
-----------------------------------

.. warning::
  Transformations in this subsection are still WIP and will be available soon.

.. autoclass:: opvious.OmitConstraints
   :noindex:
   :members:
   :show-inheritance:

.. autoclass:: opvious.OmitObjectives
   :noindex:
   :members:
   :show-inheritance:


Enforcing a minimum objective level
-----------------------------------

.. warning::
  Transformations in this subsection are still WIP and will be available soon.

.. autoclass:: opvious.ConstrainObjective
   :noindex:
   :members:
   :show-inheritance:


Transformation patterns
***********************

This section highlights a few common patterns built from the above base 
transformations.

Detecting infeasibilities
-------------------------

.. code-block:: python

  await client.run_solve(
      # ...
      transformations=[
          opvious.OmitObjectives(), # Drop existing objectives
          opvious.RelaxConstraints(), # Relax all constraints
      ],
      strategy=opvious.SolveStrategy.equally_weighted_sum("MINIMIZE"),
  )


Solution smoothing
------------------

.. code-block:: python

  await client.run_solve(
      # ...
      transformations=[
          opvious.PinVariables(["production"]),
          opvious.RelaxConstraints(["production_isPinned"]),
      ],
      strategy=opvious.SolveStrategy({
          "production_isPinned_minimizeDeficit": 100, # Smoothing factor
          # ... Other objectives
      }),
  )


Weighted distance multi-objective optimization
----------------------------------------------

.. code-block:: python

  await client.run_solve(
      # ...
      transformations=[
          # Set the target values
          opvious.ConstrainObjective("foo", min_value=5),
          opvious.ConstrainObjective("bar", max_value=10),
          # Replace the original objectives
          opvious.OmitObjectives(["foo", "bar"]),
          opvious.RelaxConstraints([
              "foo_isAtLeastMinimum",
              "bar_isAtMostMaximum",
          ]),
      ],
      strategy=opvious.SolveStrategy.equally_weighted_sum(),
  )
