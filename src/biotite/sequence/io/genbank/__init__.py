# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading/writing information
(especially sequence features) from/to files in the *GenBank*
and *GenPept* format.
"""

__name__ = "biotite.sequence.io.genbank"
__author__ = "Patrick Kunzmann"

from .annotation import *
from .file import *
from .metadata import *
from .sequence import *
