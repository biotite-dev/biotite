.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Development guide
=================

Writing code
------------

Python version and interpreter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Biotite is made for usage with Python 3.4 and upwards. Therefore no
compatibility hacks for Python 2.x are necessary. Furthermore, this package is
currently made for use with CPython. Support for PyPy might be added someday.

Code style
^^^^^^^^^^
Biotite is in compliance with PEP 8. The maximum line length is 79 for
code lines and 72 for docstring and comment lines. An exception is made for
docstring lines, if it is not possible to use a maximum of 72 characters
(e.g. tables), and for code example lines, where the actual code may take
up to 79 characters.

Dependencies
^^^^^^^^^^^^
Biotite currently depends on `numpy`, `requests` and `msgpack`.
The usage of these packages is not only allowed but even encouraged. Further
packages might be added to the depedencies in the future, so if you need a
specific package, you might open an issue on GitHub. But keep in mind, that a
simple installation process is a central aim of Biotite, so the new dependency
should neither be hard to install on any system nor be poorly supported.

Another approach is adding your special dependency to the list of extra
requirements in ``install.rst``. In this case, put the import statement for the
dependency directly into the function or class, rather than module level,
to ensure that the package is not required for any other functionality or for
building the API documentation.

If your added code has a dependency that is too special, consider publishing it
as extension package.

Code efficiency
^^^^^^^^^^^^^^^
Although convenient usage is a primary aim of Biotite, code efficiency
plays also an important role. Therefore time consuming tasks should be
C-accelerated, if possible.
The most convenient way to achieve this, is using *NumPy*.
In cases the problem is not vectorizable, writing modules in *Cython* are the
preferred way to go. Writing pure C-extensions is discouraged due to the bad
readability.
And anyway, *Cython* is *so* much better...

Docstrings
^^^^^^^^^^
Biotite uses
`numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_
formatted docstrings for its documentation.
The docstrings can be interpreted by *Sphinx* via the *numpydoc* extension.
All publicly accessible attributes must be fully documented, this includes
functions, classes, methods, instance and class variables and the
``__init__`` modules.
The ``__init__`` module documentation summarizes the content of the entire
subpackage, since the single modules are not visible to the user.
Consequently, all other modules do not need to be fully documented, one or
two short sentences are sufficient.
In the class docstring, the class itself is described and the constructor is
documented. The publicly accessible instance variables are documented under the
`Attributes` headline, while class variables are documented in their separate
docstrings. Methods do not need to be summarized in the class docstring.

Module imports
^^^^^^^^^^^^^^

In Biotite, the user imports packages in conrast to single modules
(similar to *NumPy*). In order for that to work, the ``__init__.py`` file
of each Biotite subpackage needs to import all of its modules,
whose content is publicly accessible, in a relative manner.

.. code-block:: python

   from .module1 import *
   from .module2 import *

Import statements should be the only statements in a ``__init__.py`` file.

In case a module needs functionality from another subpackage of Biotite,
use a relative import. This import should target the module directly and not
the package. So import statements like the following are totally OK:

.. code-block:: python

   from ...package.subpackage.module import foo

In order to prevent namespace pollution, all modules must define the `__all__`
variable with all publicly accessible attributes of the module.

When using Biotite internal imports, always use relative imports. Otherwise
in-development testing (see below) is not possible.



Writing the documentation
-------------------------

Any documentation apart from the API reference is placed in the ``doc``
folder.
Biotite uses *Sphinx* for building its documentation and therefore is based
on *reStructuredText* files.
The line length of these ``*.rst`` files is also limited to
79 characters, with the exceptions already mentioned above. 

Updating the tutorial
^^^^^^^^^^^^^^^^^^^^^

When adding new content for broad audience, it is appreciated to update the
tutorial pages (``doc/tutorial``) as well.
The corresponding figures are put into ``doc/static/assets/figures``.

Contributing examples
^^^^^^^^^^^^^^^^^^^^^

Do you have an application of Biotite and you want to share it with the world?
Then the example gallery is the way to go.
For gallery generation the package `sphinx-gallery` is used. Please refer to
its documentation for further information script formatting.
The example scripts are placed in ``doc/examples/scripts``.



Code testing
------------

In-development tests
^^^^^^^^^^^^^^^^^^^^

For simple tests of your code, you are free to use a ``test.py`` file in the
top-level directory since this file is ignored in the ``.gitignore`` file.
Remember you have to have to use relative imports, as long as you do not want
to build and install the package after each small code change. Therefore the
import statements in ``test.py`` will look similar to this:

.. code-block:: python

   import src.biotite
   import src.biotite.sequence as seq
   import src.biotite.structure as struc
   ...

If you are writing or using an extension module in Cython, consider using
`pyximport` at the beginning of ``test.py``.

.. code-block:: python

   import pyximport
   pyximport.install()

Unit tests
^^^^^^^^^^

In order to check if your new awesome code breaks anything in Biotite,
you should run unit tests before you open a pull request. To achieve that,
run the following command in the top-level directory.

.. code-block:: python

   python setup.py test

Running unit test requires the `pytest` framework.

Adding your own unit tests for your new module (if possible), is appreciated.
The unit tests are found in the ``tests`` folder (big surprise!). If there
is already an appropriate module for you, then just add your own test function
to it. If not, create your own module and put your test function into it.



Code deployment
---------------

The binary distribution and the source distribution are created with
the following commands, respectively:

.. code-block:: python

   python setup.py bdist_wheel
   python setup.py sdist

The source distribution is pure *Python*, hence *Cython* modules cannot be used
with it.

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Sphinx documentation is created using

.. code-block:: python

   python setup.py build_sphinx

in the top-level directory. The HTML output can be found under
``doc/_build/html``.



Required packages
-----------------

The following packages are required for the complete build process:
   
   - *numpy*
   - *matplotlib*
   - *requests*
   - *msgpack*
   - *mdtraj*
   - *Cython*
   - *Sphinx*
   - *numpydoc*
   - *sphinx-gallery*

Furthermore the following software must be installed:

   - *MUSCLE*
   - *MAFFT*
   - *Clustal Omega*



Extension packages
------------------

Biotite extension packages are Python packages that provide further
functionality for Biotite objects (`AtomArray`, `Sequence`, etc.)
or offer objects that build up on these ones.

There can be good reasons why one could choose to publish code as extension
package instead of contributing it directly to the Biotite project:
   
   - Independent development
   - An incompatible license
   - The code's use cases are too specialized
   - Unsuitable dependencies
   - Acceleration by C/C++ code (in contrast to Cython code)

If your code fulfills the following conditions

   - extends Biotite functionality
   - is documented
   - is unit tested

you can contact the maintainer or open an issue to ask for official
acceptance as extension package.

The extension packages are displayed on the *Extensions* page in the
documentation.