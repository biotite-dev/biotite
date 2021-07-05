# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This subpackage provides functionality for sequence alignments.

The two central classes involved are :class:`SubstitutionMatrix` and
:class:`Alignment`:

Every function that performs an alignment requires a
:class:`SubstitutionMatrix` that provides similarity scores for each
symbol combination of two alphabets (usually both alphabets are equal).
The alphabets in the :class:`SubstitutionMatrix` must match or extend
the alphabets of the sequences to be aligned.

An alignment cannot be directly represented as list of :class:`Sequence`
objects, since a gap indicates the absence of any symbol.
Instead, the aligning functions return one or more :class:`Alignment`
instances.
These objects contain the original sequences and a trace, that describe
which positions (indices) in the sequences are aligned.
Optionally they also contain the similarity score.

The aligning functions are usually C-accelerated, reducing the
computation time substantially.

This subpackage also contains functionality for finding *k-mer* matches
between two sequences, allowing fast heuristic pairwise alignments. 
"""

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"

from .alignment import *
from .banded import *
from .kmeralphabet import *
from .kmertable import *
from .kmersimilarity import *
from .localgapped import *
from .localungapped import *
from .matrix import *
from .multiple import *
from .pairwise import *
from .statistics import *