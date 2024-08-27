# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing sequence objects
using the popular FASTA format.

This package contains the :class:`FastaFile`, which provides a
dictionary like interface to FASTA files, where the header lines are
keys and the strings containing sequence data are the corresponding
values.

Furthermore, the package contains convenience functions for
getting/setting directly :class:`Sequence` objects, rather than strings.
"""

__name__ = "biotite.sequence.io.fasta"
__author__ = "Patrick Kunzmann"

from .convert import *
from .file import *
