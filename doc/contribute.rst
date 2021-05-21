.. This source code is part of the Biotite package and is distributed
   under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
   information.

Contributing
============

As the aim of *Biotite* is to create a comprehensive library, we welcome
developers who would like to extend the package with new functionalities or
improve existing code.

The complete development workflow is hosted on
`GitHub <https://github.com/biotite-dev/biotite>`_.
This is also the place where you would post feature propositions,
questions, bug reports, etc.

If you are interested in improving *Biotite*, you feel free join our chat on
`Discord <https://discord.gg/cUjDguF>`_.
We are happy to answer questions, discuss ideas and provide mentoring for
newcomers.
Alternatively, you can also contact `<padix.key@gmail.com>`_.
A good place to find projects to start with are the
`Open Issues <https://github.com/biotite-dev/biotite/issues>`_ and
the `Project Boards <https://github.com/biotite-dev/biotite/projects>`_.

The following page explains the development guidelines in order to keep
*Biotite*'s source code consistent.


Writing code
------------

Scope
^^^^^
The scope of *Biotite* are methods that make up the backbone of
computational molecular biology. Thus, new functionalities added to
*Biotite* should be relatively general and well established.

Code of which the purpose is too special could be published as
:ref:`extension package <extensions>`.

Consistency
^^^^^^^^^^^
New functionalities should work with existing types, if applicable.
Specifically, this includes for example :class:`AtomArray`,
:class:`AtomArrayStack`, :class:`Sequence`, :class:`Annotation`
and of course :class:`ndarray`.

Python version and interpreter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
*Biotite* is made for usage with Python 3.6 and upwards.
Therefore, no compatibility hacks for Python 2.x are necessary.
Furthermore, this package is currently made for use with CPython.
Support for PyPy might be added someday.

Code style
^^^^^^^^^^
*Biotite* is in compliance with PEP 8. The maximum line length is 79 for
code lines and 72 for docstring and comment lines.
An exception is made for docstring lines, if it is not possible to use a
maximum of 72 characters (e.g. tables), and for doctest style lines, where the
actual code may take up to 79 characters.

Dependencies
^^^^^^^^^^^^
*Biotite* currently depends on `numpy`, `requests`, `msgpack`.
The usage of these packages is not only allowed but even encouraged.
Further packages might be added to the dependencies in the future, so if you
need a specific package, you might open an issue on GitHub.
But keep in mind, that a simple installation process is a central aim of
*Biotite*, so the new dependency should neither be hard to install on any
system nor be poorly supported.

Another approach is adding your special dependency to the list of extra
requirements in ``install.rst``.
In this case, put the import statement for the dependency directly into the
function or class, rather than module level, to ensure that the package is not
required for any other functionality or for building the API documentation.

If your added code has a dependency that is too special, consider publishing
the code as :ref:`extension package <extensions>`.

Code efficiency
^^^^^^^^^^^^^^^
Although convenient usage is a primary aim of *Biotite*, code efficiency
plays also an important role.
Therefore time consuming tasks should be C-accelerated, if possible.
The most convenient way to achieve this, is using *NumPy*.
In cases the problem is not vectorizable, writing modules in *Cython* are the
preferred way to go.
Writing pure C-extensions is discouraged due to the bad readability.

Docstrings
^^^^^^^^^^
*Biotite* uses
`numpydoc <https://numpydoc.readthedocs.io/en/latest/>`_
formatted docstrings for its documentation.
The docstrings can be interpreted by *Sphinx* via the *numpydoc* extension.
All publicly accessible attributes must be fully documented.
This includes functions, classes, methods, instance and class variables and the
``__init__`` modules.
The ``__init__`` module documentation summarizes the content of the entire
subpackage, since the single modules are not visible to the user.
Consequently, all other modules do not need to be fully documented on the
module level, one or two short sentences are sufficient.
In the class docstring, the class itself is described and the constructor is
documented.
The publicly accessible instance variables are documented under the
`Attributes` headline, while class variables are documented in their separate
docstrings.
Methods do not need to be summarized in the class docstring.

Module imports
^^^^^^^^^^^^^^

In *Biotite*, the user imports packages in contrast to single modules
(similar to *NumPy*).
In order for that to work, the ``__init__.py`` file of each *Biotite*
subpackage needs to import all of its modules, whose content is publicly
accessible, in a relative manner.

.. code-block:: python

   from .module1 import *
   from .module2 import *

Import statements should be the only statements in a ``__init__.py`` file.

In case a module needs functionality from another subpackage of *Biotite*,
use a relative import.
This import should target the module directly and not the package.
So import statements like the following are totally OK:

.. code-block:: python

   from ...package.subpackage.module import foo

In order to prevent namespace pollution, all modules must define the `__all__`
variable with all publicly accessible attributes of the module.

When using *Biotite* internal imports, always use relative imports. Otherwise
:ref:`in-development testing <tests>` is not possible.

.. Type annotations
   ^^^^^^^^^^^^^^^^
   
   *Biotite* obligatorily uses type annotations (:PEP:`484`) for its public API.
   This enables static type checkers (e.g. *mypy*) to detect programming errors
   at compile time.
   Instead of using inline type annotations, the type hints are outsourced
   into ``*.pyi`` stub files, that exist alongside ``*.py`` files with the same
   module name.
   Although, *NumPy* does not support type hints yet, the `ndarray` type is still
   used in type annotations



Writing the documentation
-------------------------

Any documentation apart from the API reference is placed in the ``doc``
folder.
*Biotite* uses *Sphinx* for building its documentation and therefore the
documentation is based on *reStructuredText* files.
The line length of these ``*.rst`` files is also limited to
79 characters, with the exceptions already mentioned above. 

Contributing examples
^^^^^^^^^^^^^^^^^^^^^

Do you have an application of *Biotite* and you want to share it with the
world?
Then the example gallery is the way to go.
For gallery generation the package *sphinx-gallery* is used.
Please refer to its
`documentation <http://sphinx-gallery.readthedocs.io/en/latest/>`_
for further information on script formatting.
The example scripts are placed in ``doc/examples/scripts``.

Static images and molecular visualizations
""""""""""""""""""""""""""""""""""""""""""

In addition to *Matplotlib* plots, the *Biotite* example gallery can also
show molecular visualizations, via the *PyMOL* software, and static images.

Static images can be included by adding the following comment in the
corresponding code block:

.. code-block:: python

   # biotite_static_image = <name_of_the_image>.png

The image file must be stored in the same directory as the example script.

|

To visualize images using *PyMOL*, the
`Ammolite <https://ammolite.biotite-python.org/>`_ package is required.
Please make sure to use open-source *PyMOL* to avoid licensing issues.

Let's assume you have an example script `<example_name>.py`.
The visualization is initiated by adding the comment line

.. code-block:: python

   # Visualization with PyMOL..

in the code block where you want show the visualization.
Then the visualization script ``<example_name>_pymol.py`` is executed, which
can use the global variables from the example script and the special
``__image_destination__`` variable.
``__image_destination__`` is a string representing the path to the output image
file.
The PyMOL visualization can be saved to this file with e.g.

```python
ammolite.cmd.png(__image_destination__)
```

The rendered image is saved in the directory of the example script as
``<example_name>.png`` and is added to version control.
The visualization script is only executed, if the rendered image does not
exist, yet.
The traceback of errors in the visualization script are printed, if
``sphinx-build`` is run in verbose (``-v``) mode.
An example of this can be seen in the
``doc/examples/structure/contact_sites.py`` example.


Updating the tutorial
^^^^^^^^^^^^^^^^^^^^^

When adding new content for broad audience, it is appreciated to update the
tutorial pages (``doc/tutorial/src``) as well.
The tutorial uses functionality from ``sphinx-gallery`` to generate
the tutorial from example scripts.
This has the advantage that the output of code snippets is not static but
dynamically generated based on the current state of the *Biotite* source
code.
Consequently, the same script formatting as for the example gallery is
required.
Figures that cannot be dynamically generated are put into
``doc/static/assets/figures``.

Structuring the API reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each  *Biotite* subpackage has a dedicated reference page, describing
its classes and functions.
The categories and classes/functions that are assigned to it can be set
in ``apidoc.json``.
Classes/functions that are not assigned to any category are placed in
the 'Miscellaneous' category or, if no class/function is assigned,
in the 'Content' category.



Code testing
------------

.. _tests:

In-development tests
^^^^^^^^^^^^^^^^^^^^

For simple tests of your code, you are free to use a ``test.py`` file in the
top-level directory since this file is ignored in the ``.gitignore`` file.
Remember you have to have to use relative imports, as long as you do not want
to build and install the package after each small code change.
Therefore, the *import* statements in ``test.py`` will look similar to this:

.. code-block:: python

   import src.biotite
   import src.biotite.sequence as seq
   import src.biotite.structure as struc
   ...

Alternatively, you can install *Biotite* in development mode via
`pip install -e .`.

If you are writing or using an extension module in Cython, consider using
`pyximport` at the beginning of ``test.py``.

.. code-block:: python

   import pyximport
   pyximport.install()

Unit tests
^^^^^^^^^^

In order to check if your new awesome code breaks anything in *Biotite*,
you should run unit tests before you open a pull request.
To achieve that, install the package and run ``pytest`` in the top-level
directory.

.. code-block:: console

   $ pip install .
   $ pytest

Adding your own unit tests for your new module (if possible), is appreciated.
The unit tests are found in the ``tests`` folder (big surprise!).
If there is already an appropriate module for you, then just add your own test
function to it.
If not, create your own module and put your test function into it.



Code deployment
---------------

The binary distribution and the source distribution are created with
the following commands, respectively:

.. code-block:: console

   $ python setup.py bdist_wheel
   $ python setup.py sdist

Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Sphinx documentation is created using

.. code-block:: console

   $ pip install -e .
   $ sphinx-build doc build/doc

in the top-level directory.
The building process can take a while, since the code from the tutorial
and the example gallery is executed.
In order to omit building the tutorial and gallery, type

.. code-block:: console

   $ sphinx-build -D plot_gallery=0 doc build/doc

instead.



Required packages
-----------------

The following packages are required for the complete build process:
   
   - *numpy*
   - *scipy*
   - *networkx*
   - *matplotlib*
   - *requests*
   - *msgpack*
   - *mdtraj*
   - *cython*
   - *pytest*
   - *sphinx*
   - *numpydoc*
   - *sphinx-gallery*

Furthermore, the following software must be installed:

   - *MUSCLE*
   - *MAFFT*
   - *Clustal Omega*
   - *DSSP*

If you use the *Conda* package manager, there is a shortcut:
Cou can download a *Conda* environment from
`here <http://raw.githubusercontent.com/biotite-dev/biotite/master/environment.yml>`_,
that contains all of these requirements.
How to create and activate the environment from the ``environment.yml`` file,
is explained in the
`conda documentation <http://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_.


.. _extensions:

Extension packages
------------------

*Biotite* extension packages are Python packages that provide further
functionality for *Biotite* objects (:class:`AtomArray`, :class:`Sequence`,
etc.)
or offer objects that build up on these ones.

There can be good reasons why one could choose to publish code as extension
package instead of contributing it directly to the *Biotite* project:
   
   - Independent development
   - An incompatible license
   - The code's use cases are too specialized
   - Unsuitable dependencies
   - Acceleration by C/C++ code (in contrast to Cython code)

If your code fulfills the following conditions

   - extends *Biotite* functionality
   - is documented
   - is well tested

you can contact the *Biotite* maintainer or open an issue
to ask for official acceptance as extension package.

The current extension packages are displayed on the
:doc:`extensions section <extensions>`
in the
documentation.