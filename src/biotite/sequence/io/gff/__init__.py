# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing sequence features in the
GFF3 format.

It provides the `GFFFile` class, a low-level line-based interface to
this format, and high-level functions for extracting `Feature` and
`Annotation` objects.
"""

__author__ = "Patrick Kunzmann"

from .file import *
from .convert import *