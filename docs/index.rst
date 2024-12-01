.. default-role:: code

Opvious SDK
===========

A Python SDK for solving linear, mixed-integer, and quadratic optimization 
models with the `Opvious platform`_


Highlights
----------

+ **Declarative modeling API** exportable to LaTeX
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
  The optional `aio` dependency is recommended for improved performance. It may 
  be omitted for compatibility with `Pyodide`_ environments, for example in 
  `JupyterLite`_ kernels.

You are now ready to hop on over to the :ref:`Overview` section!


Contents
---------

.. toctree::
   :maxdepth: 1

   overview
   modeling
   transformations
   strategies
   api-reference


External resources
------------------

+ `Examples repository`_
+ `GitHub repository`_
+ `PyPI entry`_


.. _Opvious platform: https://www.opvious.io
.. _API access token: https://hub.cloud.opvious.io/authorizations
.. _pandas: https://pandas.pydata.org
.. _Pyodide: https://pyodide.org
.. _JupyterLite: https://jupyterlite.readthedocs.io
.. _PyPI entry: https://pypi.python.org/pypi/opvious/
.. _GitHub repository: https://github.com/opvious/sdk.py
.. _Examples repository: https://github.com/opvious/examples
