# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This subpackage is used for reading and writing an `AtomArray` or
`AtomArrayStack` using the internal NPZ file format. This binary format
is used to store `numpy` arrays. Since atom arrays and stacks are
completely built on `numpy` arrays, this format is preferable for
Biopython internal usgae due to fast I/O operations and preservation
of all atom annotation arrays.
"""

from .file import *