.. default-role:: code

Optimization client
===================

The SDK's functionality is available via :class:`.Client` instances, which
provide a high-level interface to the underlying Opvious API.

.. contents:: Table of contents
   :local:
   :backlinks: none


Creating a client
*****************

The recommended way to create clients is via :meth:`.Client.from_environment`. 
This will automatically authenticate the client using the `OPVIOUS_TOKEN` 
environment variable:

.. code-block:: python

  client = opvious.Client.from_environment()

It's also possible to create a client directly from an API token via 
:meth:`.Client.from_token`.


Solving optimization models
***************************

The client exposes two distinct ways of solving optimization problems, described 
below:

+ Direct :ref:`solves <Solves>`, which allow finding solutions in real-time
+ Queued :ref:`attempts <Attempts>`, which support larger data sizes

Both require an existing model :ref:`specification <Specifications>`.

Specifications
--------------

Opvious enforces a clear separation between a model's specification (variable 
types, constraint definitions, etc.) and data. Specifications are created 
directly from their mathematical representation in LaTeX, separately from this 
SDK.

.. note::
  Refer to the `platform documentation <https://docs.opvious.io>`_ for 
  information on how to write a specification.

This SDK instead provides utilities for reading specifications from various 
sources, listed below.

.. autoclass:: opvious.FormulationSpecification
   :noindex:
   :members:

.. autoclass:: opvious.LocalSpecification
   :noindex:
   :members:

.. autoclass:: opvious.RemoteSpecification
   :noindex:
   :members:

.. autoclass:: opvious.InlineSpecification
   :noindex:
   :members:


Solves
------

Solves are run with the client's :meth:`.Client.run_solve` method.

.. note::
  In most environments, solves will stream progress notifications back to the 
  client. This allows for real-time updates on the ongoing solve (current 
  optimality gap, latest epsilon constraint added, etc.).
  You can view them by enabling `INFO` or `DEBUG` logging, for example:

  .. code-block:: python

    import logging

    logging.basicConfig(level=logging.INFO)


Attempts
--------


Attempts are started with the client's :meth:`.Client.start_attempt` method.

TODO

Inspecting models
*****************


TODO
