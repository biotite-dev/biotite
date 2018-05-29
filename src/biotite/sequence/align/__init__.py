# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides functionality for sequence alignemnts.

The two central classes involved are `SubstitutionMatrix` and
`Àlignment`:

Every function that performs an alignment requires a
`SubstitutionMatrix` that provides similarity scores for each symbol
combination of two alphabets (usually both alphabets are equal).
The alphabets in the `SubstitutionMatrix` must match or extend the
alphabets of the sequences to be aligned.

After alignment one or more `Alignment` instances are returned.
These objects contain the original sequences, a trace
(indices of aligned) and the similarity score.

The alignment functions are usually C-accelerated, reducing the
computation time substantially.
"""

__author__ = "Patrick Kunzmann"

from .alignment import *
from .pairwise import *
from .matrix import *