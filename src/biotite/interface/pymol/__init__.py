# ruff: noqa: F401

# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This package enables the transfer of structures from *Biotite* to
`PyMOL <https://pymol.org/>`_ for visualization and vice versa,
via *PyMOL*'s Python API:

- Import :class:`AtomArray` and :class:`AtomArrayStack` objects into *PyMOL* -
  without intermediate structure files.
- Convert *PyMOL* objects into :class:`AtomArray` and :class:`AtomArrayStack`
  instances.
- Use *Biotite*'s boolean masks for atom selection in *PyMOL*.
- Display images rendered with *PyMOL* in *Jupyter* notebooks.

Launching PyMOL
---------------

Library mode
^^^^^^^^^^^^
The recommended way to invoke *PyMOL* in a Python script depends on whether a GUI should
be displayed.
If no GUI is required, we recommend launching a *PyMOL* session in library mode.

.. code-block:: python

    import biotite.interface.pymol as pymol_interface

    pymol_interface.launch_pymol()

Usually launching *PyMOL* manually via :func:`launch_pymol()` is not even necessary:
When *Biotite* requires a *PyMOL* session, e.g. for creating a *PyMOL* object or
invoking a command, and none is already running, *PyMOL* is automatically started in
library mode.

GUI mode
^^^^^^^^
When the *PyMOL* GUI is necessary, the *PyMOL* library mode is not available.
Instead *PyMOL* can be launched in interactive (GUI) mode:

.. code-block:: python

    import biotite.interface.pymol as pymol_interface

    pymol_interface.launch_interactive_pymol("-qixkF", "-W", "400", "-H", "400")

:func:`launch_interactive_pymol()` starts *PyMOL* using the given command line options,
reinitializes it and sets necessary parameters.

After that, the usual *PyMOL* commands and the other functions from
*Biotite* are available.

Note that the *PyMOL* window will stay open after the end of the script.
This can lead to issues when using interactive Python (e.g. *IPython*):
The *PyMOL* window could not be closed by normal means and a forced termination might be
necessary.
This can be solved by using *PyMOL*'s integrated command line for executing Python.

Launching PyMOL directly
^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    This is not the recommended way to use *PyMOL* in the context of
    :mod:`biotite.interface.pymol`.
    Usage is at your own risk.

You can also launch *PyMOL* directly using the *PyMOL* Python API, that
:func:`launch_pymol()` and :func:`launch_interactive_pymol()` use internally.
In this case, it is important to call :func:`setup_parameters()` for setting
parameters that are necessary for *Biotite* to interact properly with *PyMOL*.
Furthermore, the ``pymol_instance`` parameter must be set the first time
a :class:`PyMOLObject` is created to inform *Biotite* about the *PyMOL* session.

.. code-block:: python

    from pymol2 import PyMOL
    import biotite.interface.pymol as pymol_interface

    pymol_app = PyMOL()
    pymol_app.start()
    pymol_interface.setup_parameters(pymol_instance=pymol_app)
    cmd = pymol_app.cmd

    pymol_object = pymol_interface.PyMOLObject.from_structure(
        atom_array, pymol_instance=pymol_app
    )

Common issues
-------------
As *PyMOL* is a quite complex software with a lot of its functionality written
in *C++*, sometimes unexpected results or crashes may occur under certain
circumstances.
This page should provide help in such and similar cases.

Interactive PyMOL crashes when launched on MacOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unfortunately, the *PyMOL* GUI is not supported on MacOS, as described in
`this issue <https://github.com/schrodinger/pymol-open-source/issues/97>`_.
The library mode launched by default should still work.

Interactive PyMOL crashes when launched after usage of Matplotlib
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Interactive *PyMOL* will crash, if it is launched after a *Matplotlib* figure
is created. This does not happen in the object-oriented library mode of
*PyMOL*.
Presumably the reason is a conflict in the *OpenGL* contexts.

Example code that leads to crash:

.. code-block:: python

  import matplotlib.pyplot as plt
  import biotite.interface.pymol as pymol_interface

  figure = plt.figure()
  pymol_interface.launch_interactive_pymol()

'cmd.png()' command crashes in pytest function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``pytest`` executes the test functions via ``exec()``, which might lead to the crash.
Up to now the only way to prevent this, is not to test the ``png()`` command
in pytest.

Launching PyMOL for the first time raises DuplicatePyMOLError
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For example the code snippet

.. code-block:: python

  import biotite.interface.pymol import cmd, launch_pymol

  launch_pymol()

raises

.. code-block:: python

  biotite.interface.pymol.DuplicatePyMOLError: A PyMOL instance is already running

The reason:

If ``from biotite.interface.pymol import pymol``
or ``from biotite.interface.pymol import cmd`` is called, *PyMOL* is already launched
upon import in order to make the ``pymol`` or ``cmd`` attribute available.
Subsequent calls of :func:`launch_pymol()` or
:func:`launch_interactive_pymol()` would start a second *PyMOL* session,
which is forbidden.

To circumvent this problem do not import ``pymol`` or ``cmd`` from
``biotite.interface.pymol``, but access these attributes via ``pymol_interface.pymol``
or ``pymol_interface.cmd`` at the required places in your code.

|

*PyMOL is a trademark of Schrodinger, LLC.*
"""

__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"

from biotite.interface.version import require_package

require_package("pymol")

from .cgo import *
from .convert import *
from .display import *
from .object import *
from .shapes import *

# Do not import expose the internally used 'get_and_set_pymol_instance()'
from .startup import (
    DuplicatePyMOLError,
    launch_interactive_pymol,
    launch_pymol,
    reset,
    setup_parameters,
)


# Make the PyMOL instance accessible via `biotite.interface.pymol.pymol`
# analogous to a '@property' of a class, but on module level instead
def __getattr__(name):
    from .startup import get_and_set_pymol_instance

    if name == "pymol":
        return get_and_set_pymol_instance()
    elif name == "cmd":
        return __getattr__("pymol").cmd
    elif name in list(globals().keys()):
        return globals()["name"]
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return list(globals().keys()) + ["pymol", "cmd"]
