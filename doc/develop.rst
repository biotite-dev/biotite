Development guidelines
======================

Writing code
------------

Python version and interpreter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Biopython 2.0 is made for usage with Python 3.6 and upwards. Therefore no
compatibility hacks for Python 2.x are necessary. Furthermore this package is
currently made for use with CPython. Support for PyPy might be added someday.

Code style
^^^^^^^^^^
Biopython 2.0 is in compliance with PEP 8. The maximum line length is 79 for
code lines and 72 for docstring and comment lines. An exception is made for
docstring lines, if it is not possible to use a maximum of 72 characters
(e.g. tables), and for code example lines, where the actual code may take
up to 79 characters.

Dependencies
^^^^^^^^^^^^
Biopython currently depends on `numpy`, `scipy`, `matplotlib` and `requests`.
The usage of these packages is not only allowed but even encouraged. Further
packages might be added to the depedencies in the future, so if you need a
specific package, you might start a discussion in the Biopython community.
But keep in mind, that a simple installation process is a central aim of
Biopython 2.0, so the new dependency should neither be hard to install on
any system nor be poorly supported.

Another approach is adding your special dependency to the `extras_require`
parameter in the ``setup.py``. In this case, put the import statement for the
dependency directly into the class or function, unless the dependency is
required for the entire subpackage.

Code efficiency
^^^^^^^^^^^^^^^
Although convenient usage is a primary aim of Biopython 2.0, code efficiency
plays also an important role. Therefore time consuming tasks should be
C-accelerated, if possible.
The most convenient way to achieve this, is using `numpy`.
In cases the problem is not vectorizable, writing modules in Cython are the
preferred way to go. You have to keep in mind that Biopython 2.0 is also
required to work without extension modules, so a pure Python alternative must
always be shipped, too. The way this is solved can be seen for example in
``biopython/sequence/align/align``.
Writing pure C-extension is disencouraged due to the bad readability.
And anyway, Cython is *so* much better...

Code documentation
^^^^^^^^^^^^^^^^^^
Biopython 2.0 uses
`numpydoc <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
for its documentation. The docstrings can be interpreted by *Sphinx* via the
*numpydoc* extension. All publicly accessible attributes must be fully
documented, this includes functions, classes, methods, instance and class
variables and the ``__init__`` modules.
The ``__init__`` module documentation summarizes the content of the entire
subpackage, since the single modules do not exist for the user.
Consequently, all other modules do not need to be fully documented, one or
two short sentences are sufficient.
In the class docstring, the class itself is described and the constructor is
documented. The publicly accessible instance variables are documented under the
`Attributes` headline, while class variables are documented in their separate
docstrings. Methods do not need to be summarized in the class docstring.

Any content, that is not part of the API reference is placed in the ``doc``
folder in *ReST* format. The line length of ``*.rst`` files is also limited to
79 characters, with te exceptions already mentioned above.

Module imports
^^^^^^^^^^^^^^

In Biopython 2.0, the user imports packages rather than single modules
(similar to `numpy`). In order for that to work, the ``__init__.py`` file
of each Biopython 2.0 subpackage needs to import all of its modules,
whose content is publicly accessible, in a relative manner.

.. code-block:: python

   from .module1 import *
   from .module2 import *

Import statements should be the only statements in a ``__init__.py`` file.

In case a module needs functionality from another subpackage of Biopython 2.0,
use a relative import. This import should target the module directly and not
the package. So import statements like the following are totally OK:

.. code-block:: python

   from ...package.subpackage.module import foo

In order to prevent namespace pollution, all modules must define the `__all__`
variable with all publicly accessible attributes of the module.

When using Biopython internal imports, always use relative imports. Otherwise
In-development testing (see below) is not possible.

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

   import src.biopython
   import src.biopython.sequence as seq
   import src.biopython.structure as struc
   ...

If you are writing or using an extension module in Cython, consider using
`pyximport` at the beginning of ``test.py``.

.. code-block:: python

   import pyximport
   pyximport.install()

Unit tests
^^^^^^^^^^

In order to check if your new awesome code breaks anything in Biopython,
you should run unit tests before you open a pull request. To achieve that,
run the following command in the top-level directory.

.. code-block:: python

   python setup.py test

Adding your own unit tests for your new module (if possible), is appreciated.
Biopython 2.0 uses Python's `unittest` module for this task. The unit
tests are found in the ``tests`` folder (big surprise!). If there
is already a module with an appropriate `TestCase` for you, then just add
your own test function to it. If not, create your own module and put your
test case into it. Then import the module in the corresponding ``__init__.py``
and add the case to the test suite similar to the follwong line:

.. code-block:: python

   structure_suite.addTest(loader.loadTestsFromTestCase(SuperimposeTest))

Code deployment
---------------

The binary distribution and the source distribution are created with
the following commands, respectively:

.. code-block:: python

   python setup.py bdist_wheel
   python setup.py sdist

Building the documentation
--------------------------

The Sphinx documentation is created using

.. code-block:: python

   python setup.py build_sphinx

in the top-level directory. The HTML output can be found under
``doc/_build/html``.
