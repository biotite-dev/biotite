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
