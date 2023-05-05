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


Solving optimization models
***************************

There are two distinct ways of solving optimization problems, described below:

+ Direct solves, which allow finding solutions in real-time
+ Queued attempts, which support larger data sizes

Both require an existing model specification.

Specifications
--------------


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
