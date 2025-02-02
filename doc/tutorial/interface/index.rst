:sd_hide_title: true

.. include:: /tutorial/preamble.rst

##########################
``interface`` subpackage
##########################

Connecting the ecosystem - The ``interface`` subpackage
=======================================================

.. currentmodule:: biotite.interface

In the last section we learned that :mod:`biotite.application` encapsulates entire
external application runs with subsequent calls of ``start()`` and ``join()``.
In contrast :mod:`biotite.interface` provides flexible interfaces to other Python
packages in the bioinformatics ecosystem.
Its purpose is to convert between native Biotite objects, such as :class:`.AtomArray`
and :class:`.Sequence`, and the corresponding objects in the respective interfaced
package.
Each interface is located in a separate subpackage with the same name as the
interfaced package.
For example, the interface to ``rdkit`` is placed in the subpackage
:mod:`biotite.interface.rdkit`.

.. note::

    Like in :mod:`biotite.application`, the interfaced Python packages are not
    dependencies of the ``biotite`` package.
    Hence, they need to be installed separately.

The following chapters will give you an overview of the different interfaced packages.

.. toctree::
    :maxdepth: 1
    :hidden:

    rdkit
    pymol