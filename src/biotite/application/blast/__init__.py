# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for heuristic local alignments against a large database
using BLAST.
"""

__name__ = "biotite.application.blast"
__author__ = "Patrick Kunzmann"

from .webapp import *
from .alignment import *