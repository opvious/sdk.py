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

The recommended way to create clients is via :meth:`.Client.from_environment`. 
This will automatically authenticate the client using the `OPVIOUS_TOKEN` 
environment variable:

.. code-block:: python

  client = opvious.Client.from_environment()

It's also possible to create a client directly from an API token via 
:meth:`.Client.from_token`.

.. note::
  By default all clients connect to the Opvious production cloud. Reach out to 
  us at support@opvious.io if you are interested in a different hosting 
  solution.


Formulating problems
********************

To solve an optimization problem with Opvious, you'll need to formulate it 
first. The simplest way to get started 


Finding solutions
*****************

The client exposes two distinct ways of solving optimization problems, described 
below:

+ Direct :ref:`solves <Solves>`, which allow finding solutions in real-time
+ Queued :ref:`attempts <Attempts>`, which support larger data sizes

Both require an existing model :ref:`specification <Specifications>`.


Solves
------

Solves are run with the client's :meth:`.Client.run_solve` method.

.. automethod:: opvious.Client.run_solve
   :noindex:

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

.. automethod:: opvious.Client.start_attempt
   :noindex:


Inspecting solves
*****************

.. automethod:: opvious.Client.inspect_solve_instructions
   :noindex:
