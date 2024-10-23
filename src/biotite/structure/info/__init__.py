# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for obtaining all kinds of chemical information about atoms
and residues, including masses, radii, bonds, etc.

Most information is extracted from the *Chemical Component Dictionary*
of the
`wwPDB <ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif>`_.
"""

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann, Tom David Müller"

from .atoms import *
from .bonds import *
from .ccd import *
from .groups import *
from .masses import *
from .misc import *
from .radii import *
from .standardize import *
