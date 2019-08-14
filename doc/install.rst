.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Installation
============

Requirements
------------

*Biotite* requires at least *Python* version 3.6. If you are still using
*Python* 2.7, you should `hurry up <https://pythonclock.org/>`_ with upgrading
to *Python* 3.

*Biotite* also requires the following packages:

   - **numpy**
   - **requests**
   - **msgpack**

If you are a Linux user, you should be able to install these packages simply
via *pip* (Tip: Use ``--only-binary :all:`` to ensure precompiled versions are
installed).
In case you are using Windows, I recommend installing *numpy* and
*matplotlib* via `Conda <https://conda.io/docs/>`_, or alternatively
`Anaconda <https://www.anaconda.com/download/>`_ which already contains the
aforementioned packages.

Some functions require some extra packages:

   - **mdtraj** - Required for trajetory file I/O operations.
   - **matplotlib** - Required for plotting purposes.


Install via Conda
------------------

If you want to install *Biotite* into a *Conda* environment, you can do
that directly via *Conda*:

.. code-block:: console

   $ conda install -c conda-forge biotite

This is the preferred way to install *Biotite* for *Windows* users that
use the *Anaconda* Python distribution.


Install from PyPI
-----------------

By default, *Biotite* uses *wheels* for its package distribution. Simply type

.. code-block:: console

   $ pip install biotite

If *pip* finds an appropriate *wheel* for your system configuration on *PyPI*,
it will download and install it. Congratulations, you just installed 
*Biotite*! If no fitting *wheel* is found, *pip* will fall back to the source
distribution. If you want to prevent *pip* to do that,
use the following command:

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

If you want to install your own *Biotite* build, navigate to the top-level
directory of your local *Biotite* clone (the one ``setup.py`` is in) and type
the following:

.. code-block:: console

   $ pip install .

Note that this requires a C-compiler (typically GCC) and the packages
`cython` and `wheel` to be installed.
Having the *Biotite* package always pointing to your development directory is
also possible. Type the following in the top-level directory:

.. code-block:: console

   $ pip install -e .

To generate the wheels and source distribution for upload to PyPI (most
probably you won't need that, but just in case), simply type:

.. code-block:: console

   $ python setup.py bdist_wheel
   $ python setup.py sdist

You can find the wheel and the source distribution in the ``dist`` directory
(they should be the only files there, you can't miss them).


