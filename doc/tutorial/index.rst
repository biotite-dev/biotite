:sd_hide_title: true

########
Tutorial
########

Getting started
===============
*Biotite* is a *Python* package that facilitates everyday tasks in
sequence and structure bioinformatics by providing a broad set of tools
functionalities for handling files, analyzing data and visualizing results.
This tutorial should give newcomers a quick tour through the central
functionalities of this package and how they can be used in combination.
Thus, the following chapters use rather simple examples.
If you are more interested in application of *Biotite* on real-world problems,
have a look at the :doc:`example gallery <../examples/index>`.

Installation
------------
*Biotite* is available for *pip* and *Conda* package managers.
You can install the package simply via

.. code-block:: console

   $ pip install biotite

or

.. code-block:: console

   $ conda install -c conda-forge biotite

respectively.

If the installation was successful, you should be able to import and use
*Biotite*, for example

.. jupyter-execute::

    import biotite.sequence as seq

    print(seq.ProteinSequence("BIQTITE*IS*INSTALLED"))

If you experience issues or search for other installation methods, have a look
at the :doc:`installation page <../install>`.

Overview
--------

.. currentmodule:: biotite

*Biotite* is split into 4 subpackages:

The :mod:`biotite.sequence` subpackage contains functionality for
working with sequence information of any kind.
The package contains by default sequence types for nucleotides and
proteins, but the alphabet-based implementation allows simple
integration of own sequence types, even if they do not rely on letters.
Beside the support for different file formats, the package includes general
purpose functions for sequence manipulations and a comprehensive
modular systems for sequence alignments.

The :mod:`biotite.structure` subpackage enables handling of 3D
structures of biomolecules.
Simplified, a structure is represented by *NumPy* arrays for atom coordinates
and annotations (residue names, elements, charges, etc.).
This renders operations applied to this structure representation very fast and
scales from single models to entire ensembles (e.g. molecular dynamics
trajectories).
Structures can be read and written from many popular file formats - from the
ancient *PDB* to the modern *BinaryCIF*.
The subpackage provides functionalities for filtering, measuring, editing,
superimposing structures and much more.

The :mod:`biotite.database` subpackage is all about downloading data
from biological databases, that can be subsequently used in the aforementioned
subpackages.
It allows searching for database entries by specifying and combining criteria
in a *Pythonic* way and thereby conceals the complexity of the underlying REST
API of the database.

The :mod:`biotite.application` subpackage extends the repertoire of *Biotite*'s
analysis functions with interfaces for external software.
These range from locally installed programs (e.g. *Clustal Omega*) to web
applications (e.g. *NCBI BLAST*).
The interfaces are seamless:
The input and output are sequence and structure objects, file input/output and
the command line interface is handled internally.
It is basically very similar to using normal functions.

The following chapters will take you on a journey through the functionalities
provided by the mentioned subpackages.

.. note::

    The files used in this tutorial will be stored in a temporary directory.
    So make sure to put the files you want keep somewhere else.

.. toctree::
    :maxdepth: 1
    :hidden:

    database/index
    sequence/index
    structure/index
    application/index
    interface/index