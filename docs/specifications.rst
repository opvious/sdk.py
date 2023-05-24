Specifications
==============

Opvious enforces a clear separation between a problem's specification (variable 
types, constraint definitions, etc.) and its data. Specifications can be created 
via :class:`~.modeling.Model` instances or directly from their mathematical 
representation.

.. contents:: Table of contents
   :local:
   :backlinks: none


Creating models
***************


Definitions
-----------

TODO


Fragments
---------

TODO


Importing an existing specification
***********************************

This SDK also provides utilities for loading specifications from various 
sources, listed below.

.. note::
  Refer to the `platform documentation <https://docs.opvious.io>`_ for 
  information on how to write a specification directly.


Locally
-------

.. autoclass:: opvious.LocalSpecification
   :noindex:
   :members:


From external resources
-----------------------

.. autoclass:: opvious.FormulationSpecification
   :noindex:
   :members:

.. autoclass:: opvious.RemoteSpecification
   :noindex:
   :members:


Examples
********

+ `Example repository <https://github.com/opvious/examples/tree/main/sources>`_
+ `Platform guides <https://docs.opvious.io/guides>`_
