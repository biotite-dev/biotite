# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This subpackage provides support for the the modern PDBx/mmCIF file
format. The `PDBxFile` class provides dictionary-like access to every
field in PDBx/mmCIF files. Additional utility functions allow
conversion of these dictionaries to `AtomArray` and `AtomArrayStack`
objects and vice versa.
"""

from .convert import *
from .file import *