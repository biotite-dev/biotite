# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This subpackage is used for reading and writing an `AtomArray` or
`AtomArrayStack` using the modern PDBx/mmCIF file format. Since the
format has a dictionary-like structure, this subpackage provides
a file class that allows dictionary-style indexing.
"""

from .convert import *
from .file import *