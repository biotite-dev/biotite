# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
The MOL format is used to depict atom positions and bonds for small
molecules.
This subpackage is used for reading and writing an :class:`AtomArray`
in this format.
Additionally, reading data from the SDF format, which is a wrapper
around MOL, is also supported.
"""

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"

from .convert import *
from .header import *
from .mol import *
from .sdf import *
