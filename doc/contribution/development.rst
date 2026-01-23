Writing source code
===================

Scope
-----
The scope of *Biotite* includes methods that make up the backbone of
computational molecular biology. Thus, new functionalities added to
*Biotite* should be relatively general and well established.

Code of which the purpose is too special could be published as
:ref:`extension package <extension_packages>` instead.

Consistency
-----------
New functionalities should act on the existing central classes, if applicable
to keep the code as uniform as possible.
Specifically, these include

- :class:`biotite.structure.AtomArray`,
- :class:`biotite.structure.AtomArrayStack`,
- :class:`biotite.structure.BondList`,
- :class:`biotite.sequence.Sequence` and its subclasses,
- :class:`biotite.sequence.Alphabet`,
- :class:`biotite.sequence.Annotation`,
  including :class:`biotite.sequence.Feature`
  and :class:`biotite.sequence.Location`,
- :class:`biotite.sequence.AnnotatedSequence`,
- :class:`biotite.sequence.Profile`,
- :class:`biotite.sequence.align.Alignment`,
- :class:`biotite.application.Application` and its subclasses,
- and in general :class:`numpy.ndarray`.

If you think that the currently available classes miss a central *object*
in bioinformatics, you might consider opening an issue on *GitHub* or reach
out to the maintainers.

Small *helper classes* for a functionality (for example an :class:`Enum` for a
function parameter) is also permitted, as long as it does not introduce a
redundancy with the classes mentioned above.

Python version and interpreter
------------------------------
The package supports all minor Python versions released in the last
42 months
(`NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_).
In consequence, language features that were introduced after the oldest
supported Python version are not allowed.
This time span balances the support for older Python versions as well as
the ability to use more recent features of the programming language.

Furthermore, this package is currently made for usage with CPython.
Official support for PyPy might be added someday.

Code style
----------
*Biotite* is compliant with :pep:`8` and uses `Ruff <https://docs.astral.sh/ruff/>`_ for
code formatting and linting.
The maximum line length is 88 characters.
An exception is made for docstring lines, if it is not possible to use a
maximum of 88 characters (e.g. tables and parameter type descriptions).
To make code changes ready for a pull request, simply run

.. code-block:: console

   $ ruff format
   $ ruff check --fix

and fix the remaining linter complaints.

Dependencies
------------
*Biotite* aims to rely only on a few dependencies to keep the installation
small.
However optional dependencies for a specific dependency are also allowed if
necessary.
In this case add your special dependency to the list of extra
requirements in ``install.rst``.
The import statement for the dependency should be located directly inside the
function or class, rather than module level, to ensure that the package is not
required for any other functionality or for building the API documentation.

An example for this approach are the plotting functions in
:mod:`biotite.sequence.graphics`, that require *Matplotlib*.

Code efficiency
---------------
The central aims of *Biotite* are that it is both, convenient and fast.
Therefore, the code should be vectorized as much as possible using *NumPy*.
In cases the problem cannot be reasonably or conveniently solved this way,
writing modules in `Cython <https://cython.readthedocs.io/en/latest/>`_ is the
preferred way to go.
Writing extensions directly in C/C++ is discouraged due to the bad readability.
Writing extensions in other programming languages
(e.g. in *Rust* via `PyO3 <https://pyo3.rs>`_) is currently not permitted to
keep the build process simple.

Docstrings
----------
*Biotite* uses
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_
formatted docstrings for its documentation.
These docstrings can be interpreted by *Sphinx* via the ``numpydoc`` extension.
All publicly accessible attributes must be fully documented.
This includes functions, classes, methods, instance and class variables and the
``__init__`` modules:

The ``__init__`` module documentation summarizes the content of the entire
subpackage, since the single modules are not visible to the user.
In the class docstring, the class itself is described and the constructor is
documented.
The publicly accessible instance variables are documented under the
`Attributes` headline, while class variables are documented in their separate
docstrings.
Methods do not need to be summarized in the class docstring.

The CI validates the docstrings using ``numpydoc lint``.
However, this validation sometimes also raised false positives.
Hence, to exclude a specific function/class from validation, add the name
(or regular expression) to ``tool.numpydoc_validation.exclude`` in the
``pyproject.toml``.


Module imports
--------------
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
use an absolute import as suggested by PEP 8.
This import should target the module directly and not the package to avoid
circular imports and thus an ``ImportError``.
So import statements like the following are totally OK:

.. code-block:: python

   from biotite.subpackage.module import foo

In order to prevent namespace pollution, all modules must define the `__all__`
variable with all publicly accessible attributes of the module.

Versioning
----------
Biotite adopts `Semantic Versioning <https://semver.org>`_ for its releases.
This means that the version number is composed of three parts:

- Major version: Incremented when incompatible API changes are made.
- Minor version: Incremented when a new functionality is added in a backwards
  compatible manner.
- Patch version: Incremented when backwards compatible bug fixes are made.

Note, that such backwards incompatible changes in minor/patch versions are only
disallowed regarding the *public API*.
This means that names and types of parameters and the type of the return value
must not be changed in any function/class documented in the API reference.
However, behavioral changes (especially small ones) are allowed.

Although minor versions may not remove existing functionalities, they can
deprecate them by

- marking them as deprecated via a notice in the docstring and
- raising a `DeprecationWarning` when a deprecated functionality is used.

This gives the user a heads-up that the functionality will be removed soon.
In the next major version, deprecated functionalities can be removed entirely.

.. _extension_packages:

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
   - Extensions written in a non-permitted programming language

If your code fulfills the following conditions

   - extends *Biotite* functionality
   - is documented
   - is well tested

you can open an issue to ask for addition of the package to the
:doc:`extension package page <../extensions>`.
