Biotite documentation
=====================

.. include:: introduction.rst

Subpackages
-----------
Biotite currently consists of 4 subpackages:

Sequence
""""""""
The ``sequence`` subpackage contains functionality for working with sequence
information of any kind. The package contains by default sequence types for
nucleotides and proteins, but the alphabet-based implementation allows simple
integration of own sequence types, even if they do not rely on letters.
Beside the standard I/O operations, the package includes general purpose
functions for sequence manipulations and global/local alignments.

Structure
"""""""""
The ``structure`` subpackage enables handling of 3D structures of biomolecules.
Simplified, a structure is represented by a list of atoms and their properties,
based on `ndarrays`. The subpackage includes read/write functionality for
different formats, structure filters, coordinate transformations, angle and
bond measurements, accessible surface area calculation, structure
superimposition and more.

Database
""""""""
The ``database`` subpackage is all about ddownloading data from biological
databases, including the probably most important ones: the `RCSB PDB` and the
`NCBI Entrez` database.

Application
"""""""""""
The ``application`` subpackage provides interfaces for external software.
The interfaces range from locally installed software (e.g. MSA software) to
web apps (e.g. BLAST). The speciality is that the interfaces are seamless:
You do not have to write input files and read output files, you only have to
input `Python` objects and you get `Python` objects. It is basically very
similar to using normal `Python` functions.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   
   install
   tutorial/index
   apidoc/index
   develop
   
