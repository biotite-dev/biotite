# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage is used for reading and writing sequence alignment objects
using the ClustalW format (``.aln``).

This package contains the :class:`ClustalFile`, which provides a
dictionary-like interface to ClustalW files, where sequence names are
keys and gapped sequence strings are the corresponding values.

Furthermore, the package contains convenience functions for
getting/setting directly :class:`Alignment` objects, rather than strings.
"""

__name__ = "biotite.sequence.io.clustal"
__author__ = "Biotite contributors"

from .convert import *
from .file import *
