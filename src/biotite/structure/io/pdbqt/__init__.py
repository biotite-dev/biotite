"""
This subpackage is used for reading and writing an :class:`AtomArray` or
:class:`AtomArrayStack` using the PDBQT format used by the *AutoDock*
software series.
"""

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann"

from .convert import *
from .file import *
