# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides an interface to the `OpenMM <https://openmm.org/>`_
molecular simulation toolkit.
It allows conversion between :class:`.AtomArray`/:class:`.AtomArrayStack` and
structure-related objects from *OpenMM*.
"""

__name__ = "biotite.interface.openmm"
__author__ = "Patrick Kunzmann"

from biotite.interface.version import require_package

require_package("openmm")

from .state import *
from .system import *
