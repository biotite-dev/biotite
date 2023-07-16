r"""
Introduction
============

.. currentmodule:: biotite

*Biotite* is a *Python* package for computational biologists.
It aims to provide a broad set of tools for working with different kinds
of biological data, from sequences to structures.
On the one hand side, working with *Biotite* should be computationally
efficient, with the help of the powerful packages *NumPy* and *Cython*.
On the other hand it aims for simple usability and extensibility, so
that beginners are not overwhelmed and advanced users can easily build
upon the existing system to implement their own algorithms.

*Biotite* provides 4 subpackages:
The :mod:`biotite.sequence` subpackage contains functionality for
working with sequence information of any kind.
The package contains by default sequence types for nucleotides and
proteins, but the alphabet-based implementation allows simple
integration of own sequence types, even if they do not rely on letters.
Beside the standard I/O operations, the package includes general purpose
functions for sequence manipulations and alignments.
The :mod:`biotite.structure` subpackage enables handling of 3D
structures of biomolecules.
Simplified, a structure is represented by a list of atoms and their
properties,
based on `NumPy` arrays.
The subpackage includes read/write functionality for different formats,
structure filters, coordinate transformations, angle and bond
measurements, structure
superimposition and some more advanced analysis capabilities.
The :mod:`biotite.database` subpackage is all about downloading data
from biological databases, including the probably most important ones:
the *RCSB PDB* and the *NCBI Entrez* database.
The :mod:`biotite.application` subpackage provides interfaces for
external software.
The interfaces range from locally installed software (e.g. MSA software)
to web applications (e.g. BLAST).
The speciality is that the interfaces are seamless:
You do not have to write input files and read output files, you only
have to input *Python* objects and you get *Python* objects.
It is basically very similar to using normal functions.

In the following sections you will get an overview over the mentioned
subpackages, so go and grab some tea and cookies und let us begin.

Preliminary note
----------------

The files used in this tutorial will be stored in a temporary directory.
So make sure to put the files, you want keep, somewhere else.
"""