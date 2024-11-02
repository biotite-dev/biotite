# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for converting structures to structural alphabet sequences.

Structural alphabets represent the local geometry of each residue in a structure as
symbol in a sequence.
This allows using sequence-based functionality from :mod:`biotite.sequence` on
structural data.

For each supported structural alphabet, this subpackage provides a conversion function
that converts each chain of a given structure into a :class:`Sequence` object from the
respective structural alphabet.

Note that the structural alphabets use lower-case letters as symbols, in order to
distinguish them better from the nucleotide and amino acid alphabets.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde, Patrick Kunzmann"

from .i3d import *
from .pb import *
