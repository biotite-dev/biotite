Biopython 2.0 documentation
===========================


The Biopython package provides a general set of tool for computational biology.

Biopython 2.0 is an effort to rewrite Biopython in order to bring it on to
modern scientific Python standards. The improvements of Biopython 2.0 include:

   - **Numpy integration:**
     Structures, seqeunces, etc. rely on `numpy ndarrays`, which offers
     convenient data handling and C-accelerated operations.
   - **Restructured packages:**
     Instead of a bulk of partly incoherent packages, the package structure
     and naming in Biopython 2.0 is clear and harmonized.
   - **Usage of numpydoc:**
     Biopython 2.0 uses numpydoc for all of its classes and functions.
   - **Removal of deprecated subpackages:**
     A lot of modules in Biopython 1.x are not up to data any more.
     Since Biopython 2.0 is almost a full rewrite, there are no outdated
     packages.
   - **New functionality:**
     Last but not least Biopython 2.0 adds functionality that is not included
     in the original Biopython distribution.

Subpackages
-----------
Biopython 2.0 currently consists of 4 subpackages:

Application
"""""""""""
Command line and online applications can be managed with the ``application``
subpackage.

Database
""""""""
The ``database`` subpackage provides interfaces for searching in and fetching
data from biologial databases like *NCBI Entrez* or *RCSB*.

Sequence
""""""""
The ``sequence`` subpackage contains functionality for working with sequence
information of any kind. The package contains by default sequence types for
DNA, RNA and Proteins and allows conversion between them
(transcription/translation). The alphabet-based implementation allows simple
integration of own sequence types, even if they do not rely on letters.
Beside the standard I/O operations, the the package includes general purpose
functions for sequence manipulations and global/local alignments.

Structure
"""""""""
The ``structure`` subpackage enables handling of 3D structures of biomolecules.
Simplified, a structure is represented by a list of atoms and their properties,
based on `numpy` arrays. The subpackage includes read/write functionality for
different formats, structure filters, coordinate transformations, angle and
bond measurements, accessible surface area calculation, structure
superimposition and more.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   
   install
   tutorial/index
   apidoc/index
   develop
   
