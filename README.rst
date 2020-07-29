.. image:: https://img.shields.io/pypi/v/biotite.svg
   :target: https://pypi.python.org/pypi/biotite
   :alt: Biotite at PyPI
.. image:: https://img.shields.io/pypi/pyversions/biotite.svg
   :alt: Python version
.. image:: https://img.shields.io/travis/biotite-dev/biotite.svg
   :target: https://travis-ci.org/biotite-dev/biotite
   :alt: Travis CI status

.. image:: https://www.biotite-python.org/_static/assets/general/biotite_logo_m.png
   :alt: The Biotite Project

Biotite project
===============

*Biotite* is your Swiss army knife for bioinformatics.
Whether you want to identify homologous sequence regions in a protein family
or you would like to find disulfide bonds in a protein structure: *Biotite*
has the right tool for you.
This package bundles popular tasks in computational molecular biology
into a uniform *Python* library.
It can handle a major part of the typical workflow
for sequence and biomolecular structure data:
   
   - Searching and fetching data from biological databases
   - Reading and writing popular sequence/structure file formats
   - Analyzing and editing sequence/structure data
   - Visualizing sequence/structure data
   - Interfacing external applications for further analysis

*Biotite* internally stores most of the data as *NumPy* `ndarray` objects,
enabling

   - fast C-accelerated analysis,
   - intuitive usability through *NumPy*-like indexing syntax,
   - extensibility through direct access of the internal *NumPy* arrays.

As a result the user can skip writing code for basic functionality (like
file parsers) and can focus on what their code makes unique - from
small analysis scripts to entire bioinformatics software packages.

If you use *Biotite* in a scientific publication, please cite:

| Kunzmann, P. & Hamacher, K. BMC Bioinformatics (2018) 19:346.
| `<https://doi.org/10.1186/s12859-018-2367-z>`_


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


Contribution
------------

Interested in improving *Biotite*?
Have a look at the
`contribution guidelines <https://www.biotite-python.org/contribute.html>`.
Feel free to join or community chat on `Discord <https://discord.gg/ECdDbvD>`_.