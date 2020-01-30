.. image:: https://img.shields.io/pypi/v/biotite.svg
   :target: https://pypi.python.org/pypi/biotite
.. image:: https://img.shields.io/pypi/pyversions/biotite.svg
.. image:: https://img.shields.io/travis/biotite-dev/biotite.svg
   :target: https://travis-ci.org/biotite-dev/biotite

.. image:: doc/static/assets/general/biotite_logo_m.png
   :alt: The Biotite Project

Biotite project
===============

Overview
--------

The *Biotite* package bundles popular tasks in computational biology into an
unifying library, which is easy to use and computationally efficient.
The package features

- Sequence and structure data analysis and editing functionality
- Support for common sequence and structure file formats
- Visualization capabilities
- Access to common biological databases (*RCSB PDB*, *NCBI Entrez*)
- Interfaces to external software (MSA software, *BLAST*, *DSSP*)

*Biotite*'s complete documentation is hosted at www.biotite-python.org


Installation
------------

*Biotite* requires the following packages:

   - **numpy**
   - **requests**
   - **msgpack**

Some functions require some extra packages:

   - **mdtraj** - Required for trajetory file I/O operations.
   - **matplotlib** - Required for plotting purposes.

*Biotite* can be installed via *Conda*...

.. code-block:: console

   $ conda install -c conda-forge biotite

... or *pip*

.. code-block:: console

   $ pip install biotite
