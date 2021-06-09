# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application"
__author__ = "Patrick Kunzmann"
__all__ = ["map_sequence", "map_matrix"]


import numpy as np
from ..sequence.seqtypes import ProteinSequence
from ..sequence.align.matrix import SubstitutionMatrix


def map_sequence(sequence):
    """
    Map a sequence with an arbitrary alphabet into a
    :class:`ProteinSequence`, in order to support arbitrary sequence
    types in software that can handle protein sequences. 
    """
    if len(sequence.alphabet) > len(ProteinSequence.alphabet):
        # Cannot map into a protein sequence if the alphabet
        # has more symbols
        raise TypeError(
            f"The software cannot align sequences of type "
            f"{type(sequence).__name__}: "
            f"Alphabet is too large to be converted into amino "
            f"acid alphabet"
        )
    # Mapping is done by simply taking over the sequence
    # code of the original sequence
    mapped_sequence = ProteinSequence()
    mapped_sequence.code = sequence.code
    return mapped_sequence


def map_matrix(matrix):
    """
    Map a :class:`SubstitutionMatrix` with an arbitrary alphabet into a
    class:`SubstitutionMatrix` for protein sequences, in order to support
    arbitrary sequence types in software that can handle protein
    sequences. 
    """
    if matrix is None:
        raise TypeError(
            "A substitution matrix must be provided for custom "
            "sequence types"
        )
    # Create a protein substitution matrix with the values taken
    # from the original matrix
    # All trailing symbols are filled with zeros
    old_length = len(matrix.get_alphabet1())
    new_length = len(ProteinSequence.alphabet)
    new_score_matrix = np.zeros((new_length, new_length))
    new_score_matrix[:old_length, :old_length] = matrix.score_matrix()
    return SubstitutionMatrix(
        ProteinSequence.alphabet, ProteinSequence.alphabet,
        new_score_matrix
    )