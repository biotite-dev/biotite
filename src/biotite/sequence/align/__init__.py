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

An alignment cannot be directly represented as list of `Sequence`
objects, since a gap indicates the absence of any symbol.
Instead, the aligning functions return one or more `Alignment`
instances.
These objects contain the original sequences and a trace, that describe
which positions (indices) in the sequences are aligned.
Optionally they also contain the similarity score.

The aligning functions are usually C-accelerated, reducing the
computation time substantially.
"""

__author__ = "Patrick Kunzmann"

from .alignment import *
from .pairwise import *
from .matrix import *