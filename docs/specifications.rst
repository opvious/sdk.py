Specifications
==============

Opvious enforces a clear separation between a model's specification (variable 
types, constraint definitions, etc.) and data. Specifications are created 
directly from their mathematical representation in LaTeX, separately from this 
SDK.

.. note::
  Refer to the `platform documentation <https://docs.opvious.io>`_ for 
  information on how to write a specification.

This SDK instead provides utilities for reading specifications from various 
sources, listed below.

.. contents:: Table of contents
   :local:
   :backlinks: none


Available types
***************

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


Examples
********

+ `Example repository <https://github.com/opvious/examples/tree/main/sources>`_
+ `Platform guides <https://docs.opvious.io/guides>`_
