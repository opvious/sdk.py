.. default-role:: code

Opvious SDK
===========

A Python SDK for solving linear, mixed-integer, and quadratic optimization 
models

.. note::
  Opvious is currently in beta. You can preview the SDK's functionality on small 
  datasets without registering but you will need an API key to solve larger 
  instances. You can request to join `here <https://www.opvious.io/signup>`_ or 
  by contacting us at hello@opvious.io.


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

Then generate an API access token in the `Optimization Hub`_ and set it as 
`OPVIOUS_TOKEN` environment variable. You're now ready to hop on over to the 
:ref:`Overview` section.


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
