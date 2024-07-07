# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing an :class:`AtomArray` or
:class:`AtomArrayStack` using the popular PDB format.
Since this format has some limitations, it will be completely replaced
someday by the modern PDBx format.
Therefore this subpackage should only be used, if usage of the
PDBx (CIF or BinaryCIF) format is not suitable (e.g. when interfacing
other software).
In all other cases, usage of the :mod:`pdbx` subpackage is encouraged.
"""

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann"

from .convert import *
from .file import *
