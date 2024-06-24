:sd_hide_title: true

.. include:: /tutorial/preamble.rst

#######################
``database`` subpackage
#######################

Data to work with - The ``database`` subpackage
===============================================

Biological databases are the backbone of computational biology.
The :mod:`biotite.database` subpackage provides interfaces for popular
online databases like the *RCSB PDB* or the *NCBI Entrez* database.
These interfaces strive to work as uniform as possible, across the databases.
This means, learning how to search and download data from one database should
be enough to work the other database interfaces as well.

.. toctree::
    :maxdepth: 1
    :hidden:

    entrez
    uniprot
    rcsb
    pubchem