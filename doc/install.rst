.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Installation
============

Requirements
------------
*Biotite* requires a modern *Python* installation (at least the last two years are
supported) and the following packages:

   - ``numpy``
   - ``requests``
   - ``msgpack``
   - ``networkx``

Installation of the ``biotite`` package will also automatically install these
dependencies if not already present.

Some functionalities require extra packages:

   - ``matplotlib`` - Required for plotting purposes.


Installation from PyPI
----------------------
*Biotite* is available as *wheels* for a variety of platforms
(Windows, Linux, OS X).
Simply type

.. code-block:: console

   $ pip install biotite

If *pip* finds an appropriate *wheel* for your system configuration on *PyPI*,
it will download and install it.
If no fitting *wheel* is found, *pip* will fall back to the source
distribution.
In this case, installation will take longer, because extension modules need
to be compiled first.
Note that this requires a C-compiler (typically GCC) installed on your system.

Installation via Conda
----------------------
*Biotite* is also available via Conda.

.. code-block:: console

   $ conda install -c conda-forge biotite


Installation from source
------------------------

You can also install Biotite from the
`project repository <https://github.com/biotite-dev/biotite>`_.
After cloning the repository, navigate to its top-level directory (the one
``setup.py`` is in) and type the following:

.. code-block:: console

   $ pip install .

Having the *Biotite* installation always pointing to your repository clone is
also possible.
Type the following in the top-level directory:

.. code-block:: console

   $ pip install -e .

Updating the Chemical Component Dictionary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`biotite.structure.info` subpackage contains a subset from the
`PDB Chemical Component Dictionary (CCD) <https://www.wwpdb.org/data/ccd>`_.
The repository ships a potentially outdated version of this subset.
To update this subset to the current upstream CCD version, run

.. code-block:: console

   $ python setup_ccd.py

Afterwards, install *Biotite* again.


Common issues and solutions
---------------------------

Compiler error when building Biotite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is a known problem with *GCC* compiler errors in some *Linux*
distributions (e.g. *Arch Linux*) when building *Biotite* from source.
Among other error lines the central error is the following:

.. code-block::

   unable to initialize decompress status for section .debug_info

While the exact reason for this error is still unknown, this can be fixed by
using a *GCC* installed via *Conda*:

.. code-block:: console

   $ conda install -c conda-forge c-compiler

ValueError when importing Biotite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When importing one of *Biotite*'s subpackages one of the following
errors might occur:

.. code-block::

   ValueError: numpy.ufunc size changed, may indicate binary incompatibility.
   ValueError: numpy.ndarray size changed, may indicate binary incompatibility.

The reason for this error is, that *Biotite* was built against a *NumPy*
version other than the one installed.
This happens for example when *NumPy* is updated, but *Biotite* is already
installed.
Try updating *NumPy* and *Biotite* to solve this issue.