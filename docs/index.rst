.. default-role:: code

Opvious Python SDK
==================

An SDK for solving optimization problems with the Opvious API

+ `Project homepage on GitHub`_
+ `PyPI entry`_


Installation
------------

Using `pip`:

.. code-block:: bash

  $ pip install opvious[aio]

The optional `aio` dependency is recommended for improved performance.
It may be omitted for compatibility with Pyodide environments, for example in
`JupyterLite`_ kernels:

.. code-block:: python

  import piplite
  await piplite.install('opvious')


Contents
---------

.. toctree::
   :maxdepth: 2

   reference

.. _Project homepage on GitHub: https://github.com/opvious/sdk.py
.. _PyPI entry: https://pypi.python.org/pypi/opvious/
.. _JupyterLite: https://jupyterlite.readthedocs.io
