# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing an `AtomArray` or
`AtomArrayStack` using the internal NPZ file format. This binary format
is used to store `NumPy` arrays. Since atom arrays and stacks are
completely built on `NumPy` arrays, this format is preferable for
Biotite internal usage due to fast I/O operations and preservation
of all atom annotation arrays.
"""

__author__ = "Patrick Kunzmann"

from .file import *