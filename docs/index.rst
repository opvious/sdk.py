.. default-role:: code

Opvious SDK
===========

A Python SDK for solving linear, mixed-integer, and quadratic optimization 
models


Highlights
----------

+ **Remote solves** with real-time progress notifications, no local solver 
  installation required
+ Seamless **data import/export** via native support for `pandas`
+ Flexible **multi-objective support**: weighted sums, epsilon constraints, ...
+ Built-in **debugging capabilities**: relaxations, fully annotated LP 
  formatting, ...


Getting started
---------------

First install the SDK, for example using `pip`:

.. code-block:: bash

  $ pip install opvious[aio]


.. note::
  The optional `aio` dependency is recommended for improved performance.
  It may be omitted for compatibility with `Pyodide`_ environments, for example
  in `JupyterLite`_ kernels.

Then generate an API access token in the `Optimization Hub`_ and set it as 
`OPVIOUS_TOKEN` environment variable. You're now ready to hop on over to the 
:ref:`Optimization client` section.

.. note::
  Opvious is currently in closed beta. You will need to join before you can 
  generate an API key. You can sign-up `here <https://www.opvious.io/signup>`_ 
  or by contacting us at hello@opvious.io.


Contents
---------

.. toctree::
   :maxdepth: 1

   client
   specifications
   transformations
   strategies
   api-reference


External resources
------------------

+ `Examples repository`_
+ `Platform documentation`_
+ `GitHub repository`_
+ `PyPI entry`_


.. _Optimization Hub: https://hub.beta.opvious.io/
.. _pandas: https://pandas.pydata.org
.. _Pyodide: https://pyodide.org
.. _JupyterLite: https://jupyterlite.readthedocs.io
.. _PyPI entry: https://pypi.python.org/pypi/opvious/
.. _GitHub repository: https://github.com/opvious/sdk.py
.. _Examples repository: https://github.com/opvious/examples
.. _Platform documentation: https://docs.opvious.io
