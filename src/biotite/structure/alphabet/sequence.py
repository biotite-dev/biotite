# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.alphabet.sequence"
__author__ = "Martin Larralde"
__all__ = ["StructureSequence"]

import numpy as np
from biotite.sequence.alphabet import AlphabetError, AlphabetMapper, LetterAlphabet
from biotite.sequence.sequence import Sequence


class StructureSequence(Sequence):
    """
    Representation of a structure sequence (in 3di alphabet).

    Parameters
    ----------
    sequence : iterable object, optional
        The initial protein sequence. This may either be a list or a
        string. May take upper or lower case letters. If a list is
        given, the list elements must be 1-letter state representations.
        By default the sequence is empty.
    """

    _codon_table = None
    alphabet = LetterAlphabet(
        [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
    )

    def get_alphabet(self):
        return StructureSequence.alphabet

    def __repr__(self):
        """Represent StructureSequence as a string for debugging."""
        return f'StructureSequence("{"".join(self.symbols)}")'