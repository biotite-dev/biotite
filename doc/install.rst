Install instructions
====================

*Biotite* comes in two flavors: A binary distribution with some extra
C-accelerated functions (e.g. for alignments or mmCIF parsing) and a
source distribution without the extension modules. Note that the source
distribution still has the same functionality - some operations are just a lot
slower.

Requirements
------------

*Biotite* requires at least *Python* version 3.5. If you are still using
*Python* 2.7, you should `hurry up <https://pythonclock.org/>`_ with upgrading
to *Python* 3.x.

*Biotite* also requires the following packages:

   - **numpy**
   - **matplotlib**
   - **requests**

If you are a Linux user, you should be able to install these packages simply
via *pip* (Tip: Use ``--only-binary :all:`` to ensure precompiled versions are
installed).
In case you are using Windows I recommend installing *numpy* and
*matplotlib* via `Conda <https://conda.io/docs/>`_ or alternatively
`Anaconda <https://www.anaconda.com/download/>`_ which already contains the
forementioned packages.

Some functions require some extra packages:

   - **mdtraj** - Required for trajetory file I/O operations.

Binary distribution
-------------------

*Biotite* uses *wheels* for binary package distributions. This is the
default way to install *Biotite*, therefore you can just type this:

.. code-block:: python

   pip install biotite

If *pip* finds an appropriate *wheel* for your system configuration on *PyPI*,
it will download and install it. Congratulations, you just installed 
*Biotite*! If no fitting *wheel* is found, *pip* will fall back to the
already mentioned source distribution. If you want to prevent *pip* to do that,
use the following command:

.. code-block:: python

   pip install biotite --only-binary :all:

In case there is no *wheel* available for you, but you still want the
juicy performance increase, you have to build the *wheel* on your own.
In order to do that, you first need to download the *Biotite* repository or a
*Biotite* release from GitHub.  Then open a terminal in the top-level folder
(the one, ``setup.py`` is in) and type the following:

.. code-block:: python

   python setup.py bdist_wheel

Note that this step requires a C-compiler (typically GCC) and the packages
`cython` and `wheel` to be installed. Then you navigate into the ``dist``
folder and type

.. code-block:: python

   pip install <package.whl>
   
where ``<package.whl>`` is the *wheel* file existing in the directory
(it should be the only file there, you can't miss it).

You can check if your *Biotite* distribution successfully uses
C-extensions via the `has_c_extensions()` function.

.. code-block:: python

   >>> import biotite
   >>> print(biotite.has_c_extensions())
   True

If the function returns `False` or, even worse, an exception, then something
went wrong.

Source distribution
-------------------

The source distribution, written in pure Python, should be seen as a fallback
option, which is useful in case there is either no *wheel* available for you or
the available *wheel* is not working for some reason (or you just don't like
fast code).
*pip* will automatically install the source distribution if it does not find
an appropriate *wheel*. If you want to insist on using the source distribution,
type the following command:

.. code-block:: python

   pip install biotite --no-binary :all:

Calling the `has_c_extensions()` function should now return `False`.


