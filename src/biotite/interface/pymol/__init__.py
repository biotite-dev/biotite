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

*PyMOL is a trademark of Schrodinger, LLC.*
"""

__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"


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
