# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This subpackage is used for reading and writing sequence objects
using the popular FASTA format.

This package contains the `FastaFile`, which provides a dictionary
like interface to FASTA files, where the header lines are keys and
the the strings containing sequence data are the corresponding values.

Furthermore the package contains convenience functions for
getting/setting directly `Sequence` objects, rather than strings.
"""

from .file import *
from .convert import *