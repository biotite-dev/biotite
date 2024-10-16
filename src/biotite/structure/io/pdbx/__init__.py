# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides support for the the modern PDBx file formats.
The :class:`CIFFile` class provides dictionary-like access to
every field in text-based *mmCIF* files.
:class:`BinaryCIFFile` provides analogous functionality for the
*BinaryCIF* format.
Additional utility functions allow reading and writing structures
from/to these files.
"""

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"

from .bcif import *
from .cif import *
from .component import *
from .compress import *
from .convert import *
from .encoding import *
