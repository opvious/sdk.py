.. default-role:: code

Overview
========

The SDK's functionality is available via :class:`.Client` instances, which
provide a high-level interface to the underlying Opvious API.

.. contents:: Table of contents
   :local:
   :backlinks: none


Creating a client
*****************

First, make sure you've :ref:`installed the SDK <Getting started>`. Once that's 
done, the recommended way to create clients is via :meth:`.Client.default`.

.. code-block:: python

  client = opvious.Client.default("http://localhost:8080") # Local API server
  client = opvious.Client.default(opvious.DEMO_ENDPOINT) # Demo cloud endpoint

The first argument, `endpoint`, determines the client's underlying API. For 
example the address of a self-hosted `API server 
<https://hub.docker.com/r/opvious/api-server>`_. As a convenience, we also 
provide :meth:`.Client.from_environment` which creates a client from the 
`OPVIOUS_ENDPOINT` and `OPVIOUS_TOKEN` environment variables:

.. code-block:: python

  client = opvious.Client.from_environment()


Formulating problems
********************

To solve an optimization problem, you will need to formulate it first. Opvious 
enforces a clear separation between a problem's *specification* (its abstract 
definition with variable types, constraints, objectives, etc.) and the data 
needed to solve it.

Specifications can be created:

+ via this SDK's :ref:`declarative modeling API <Modeling>`, which will 
  automatically generate the problem's mathematical representation;
+ or directly from a problem's mathematical representation in LaTeX, written 
  separately from this SDK.

Model instances
---------------

.. note::
  Refer to the :ref:`modeling page <Modeling>` for information on how to create 
  :class:`~opvious.modeling.Model` instances.

Calling any model's :meth:`~opvious.modeling.Model.specification` method will 
return its specification:

.. code-block:: python

  import opvious.modeling as om

  class SetCover(Model):
    """Sample set cover specification"""

    sets = Dimension()
    vertices = Dimension()
    covers = Parameter.indicator(sets, vertices)
    used = Variable.indicator(sets)

    @constraint
    def all_covered(self):
      for v in self.vertices:
        count = total(self.used(s) * self.covers(s, v)for s in self.sets)
        yield count >= 1

    @objective
    def minimize_used(self):
      return total(self.used(s) for s in self.sets)

  model = SetCover()
  specification = model.specification()

The returned :class:`~opvious.LocalSpecification` instances are integrated with 
IPython's rich display capabilities and will be pretty-printed within notebooks. 
For example the above specification will be displayed as:

.. math::

  \begin{align*}
    \S^d_\mathrm{sets}&: S \\
    \S^d_\mathrm{vertices}&: V \\
    \S^p_\mathrm{covers}&: c \in \{0, 1\}^{S \times V} \\
    \S^v_\mathrm{used}&: \psi \in \{0, 1\}^{S} \\
    \S^c_\mathrm{allCovered}&:
      \forall v \in V, \sum_{s \in S} \psi_{s} c_{s,v} \geq 1 \\
    \S^o_\mathrm{minimizeUsed}&: \min \sum_{s \in S} \psi_{s} \\
  \end{align*}

We recommend also using the client's 
:meth:`~opvious.Client.annotate_specification` method to validate specifications 
and highlight any errors:

.. code-block:: python

   annotated = await client.annotate_specification(specification)


Specifications
--------------

.. note::
  Refer to the `platform documentation <https://docs.opvious.io>`_ for 
  information on how to write a specification directly. You can also find 
  various sample sources in our `example repository 
  <https://github.com/opvious/examples/tree/main/sources>`_.

This SDK provides utilities for loading specifications from various locations, 
listed below.

.. autoclass:: opvious.LocalSpecification
   :noindex:
   :members:

.. autoclass:: opvious.FormulationSpecification
   :noindex:
   :members:

.. autoclass:: opvious.RemoteSpecification
   :noindex:
   :members:


Finding a solution
******************

Once you have a problem's specification, the client exposes two distinct ways of 
solving it:

+ :ref:`Live solves`, which find solutions in real-time
+ :ref:`Queued solves`, which support larger data sizes


Live solves
-----------

Solves can be run in real time with the client's :meth:`.Client.solve` method.

.. automethod:: opvious.Client.solve
   :noindex:

.. note::
  In many environments, clients can stream solve progress notifications back to 
  the client. This allows for real-time updates on the ongoing solve (current 
  optimality gap, latest epsilon constraint added, etc.).
  You can view them by enabling `INFO` or `DEBUG` logging, for example:

  .. code-block:: python

    import logging

    logging.basicConfig(level=logging.INFO)


Queued solves
-------------

Solves are queued via the client's :meth:`.Client.queue_solve` method.

.. automethod:: opvious.Client.queue_solve
   :noindex:


Debugging problems
******************

.. automethod:: opvious.Client.summarize_problem
   :noindex:

.. automethod:: opvious.Client.format_problem
   :noindex:
