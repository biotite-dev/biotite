:sd_hide_title: true

.. include:: /tutorial/preamble.rst

########################
``structure`` subpackage
########################

Going 3D - The ``structure`` subpackage
=======================================

.. currentmodule:: biotite.structure

:mod:`biotite.structure` is a *Biotite* subpackage for handling
molecular structures.
This subpackage enables efficient and easy handling of protein structure data
by representing atom attributes in *NumPy* :class:`ndarray` objects.
These atom attributes include so called *annotations*
(polypetide chain id, residue name, atom name, element, charge etc.)
and the atom coordinates.

The package contains three central types: :class:`Atom`, :class:`AtomArray` and
:class:`AtomArrayStack`.
An :class:`Atom` contains data for a single atom, an :class:`AtomArray` stores
data for an entire model and :class:`AtomArrayStack` stores data for multiple
models, where each model contains the same atoms but differs in the atom
coordinates.
Both, :class:`AtomArray` and :class:`AtomArrayStack`, store the attributes in
*NumPy*`* arrays. This approach has multiple advantages:

    - Convenient selection of atoms in a structure by using *NumPy* style
      indexing
    - Fast calculations on structures using C-accelerated :class:`ndarray`
      operations
    - Simple implementation of custom calculations

Based on the implementation using :class:`ndarray` objects, this package also
contains functions for structure analysis and manipulation.

.. Note::

    The universal length unit in *Biotite* is Å.
    This includes coordinates, distances, surface areas, etc.

.. toctree::
    :maxdepth: 1
    :hidden:

    atoms
    io
    filter
    bonds
    editing
    measurement
    segments
    nucleotide
    alphabet
