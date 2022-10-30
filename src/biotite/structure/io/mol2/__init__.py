# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
The MOL format is used to depict atom positions and bonds for small
molecules.
This subpackage is used for reading and writing an :class:`AtomArray` or
an :class:`AtomArrayStack` for a file containing multiple models
in this format.
"""

__name__ = "biotite.structure.io.mol2"
__author__ = "Benjamin E. Mayer"

from .file import *
from .convert import *
