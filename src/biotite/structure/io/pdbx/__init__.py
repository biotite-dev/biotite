# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides support for the the modern PDBx/mmCIF file
format. The `PDBxFile` class provides dictionary-like access to every
field in PDBx/mmCIF files. Additional utility functions allow
conversion of these dictionaries to `AtomArray` and `AtomArrayStack`
objects and vice versa.
"""

__author__ = "Patrick Kunzmann"

from .convert import *
from .file import *