.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Installation
============

Requirements
------------

*Biotite* requires at least *Python* version 3.7 and the following packages:

   - **numpy**
   - **requests**
   - **msgpack**
   - **networkx**

If you are a Linux user, you should be able to install these packages simply
via *pip* (Tip: Use ``--only-binary :all:`` to ensure precompiled versions are
installed).
In case you are using Windows, it is recommended to install *numpy* and
*matplotlib* via `Conda <https://conda.io/docs/>`_, or alternatively
`Anaconda <https://www.anaconda.com/download/>`_ which already contains the
aforementioned packages.

Some functions require some extra packages:

   - **mdtraj** - Required for trajectory file I/O operations.
   - **matplotlib** - Required for plotting purposes.


Install via Conda
------------------

For *Conda* users, for example *Windows* users who use the *Anaconda* Python
distribution, the simplest way for installing *Biotite* is

.. code-block:: console

   $ conda install -c conda-forge biotite


Install from PyPI
-----------------

By default, *Biotite* uses *wheels* for its package distribution.
Simply type

.. code-block:: console

   $ pip install biotite

If *pip* finds an appropriate *wheel* for your system configuration on *PyPI*,
it will download and install it.
Congratulations, you just installed *Biotite*!
If no fitting *wheel* is found, *pip* will fall back to the source
distribution.
If you want to prevent *pip* from doing that, use the following command:

.. code-block:: console

  $ pip install biotite --only-binary :all:

The source distribution can be used if there is no *wheel* available for you or
you want to compile the package on your own for other reasons:

.. code-block:: console

   $ pip install biotite --no-binary :all:

Note that installing from source distribution requires a C-compiler
(typically GCC).


Install from source
-------------------

You can also install Biotite from the
`project repository <https://github.com/biotite-dev/biotite>`_.
After cloning the repository, navigate to its top-level directory (the one
``setup.py`` is in) and type the following:

.. code-block:: console

   $ pip install .

Having the *Biotite* package always pointing to your directory containing the
repository is also possible.
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