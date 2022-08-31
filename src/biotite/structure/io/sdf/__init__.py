# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
The SDF format is used to depict atom positions and bonds for small
molecules as well as multiple meta informations. 
Also there can be multiple models within one file.
This subpackage leans on the ctab modle implemented for the MOL file
but also produces a dict containing the meta information.
"""

__name__ = "biotite.structure.io.sdf"
__author__ = "Benjamin E. Mayer"

from .file import *
from .convert import *
