# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
The :class:`MSAInput` helper shared by multiple sequence alignment
applications.

It bundles the input preparation, FASTA writing and output parsing that
each MSA :class:`Application` needs, instead of an intermediate
``MSAApp`` base class.
"""

from __future__ import annotations

__name__ = "biotite.application_v2"
__author__ = "Patrick Kunzmann"
__all__ = ["MSAInput", "resolve_gap_penalty"]

import numbers
from collections import OrderedDict
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import IO, Literal
import numpy as np
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.sequence.sequence import Sequence


@dataclass(frozen=True)
class MSAInput:
    """
    The sequences and substitution matrix prepared for an MSA run.

    Attributes
    ----------
    seqtype : {'nucleotide', 'protein'}
        The sequence type the software is run with.
        A mapped custom sequence type masquerades as ``'protein'``.
    sequences : list of Sequence
        The sequences that are actually written to the input file.
        These may be mapped protein sequences.
    matrix : SubstitutionMatrix or None
        The substitution matrix that is actually written to the input
        file, or None if no matrix is used.
    original_sequences : list of Sequence
        The original input sequences, used to build the resulting
        :class:`Alignment`.
    """

    seqtype: Literal["nucleotide", "protein"]
    sequences: list[Sequence]
    matrix: SubstitutionMatrix | None
    original_sequences: list[Sequence]

    @staticmethod
    def from_input(
        sequences: SequenceABC[Sequence],
        matrix: SubstitutionMatrix | None,
        supports_nucleotide: bool,
        supports_protein: bool,
        supports_custom_nucleotide_matrix: bool,
        supports_custom_protein_matrix: bool,
    ) -> MSAInput:
        """
        Validate the input and determine how the sequences and
        substitution matrix are passed to the MSA software.

        Sequences with an exotic alphabet are mapped onto protein
        sequences, if the software supports protein sequences with a
        custom substitution matrix.

        Parameters
        ----------
        sequences : iterable object of Sequence
            The sequences to be aligned.
        matrix : SubstitutionMatrix or None
            A custom substitution matrix.
        supports_nucleotide, supports_protein : bool
            Whether the software supports nucleotide or protein
            sequences, respectively.
        supports_custom_nucleotide_matrix, supports_custom_protein_matrix : bool
            Whether the software supports custom substitution matrices
            for nucleotide or protein sequences, respectively.

        Returns
        -------
        input : MSAInput
            The prepared input.
        """
        sequences = list(sequences)
        if len(sequences) < 2:
            raise ValueError("At least two sequences are required")
        alphabet = sequences[0].get_alphabet()
        for sequence in sequences:
            if sequence.get_alphabet() != alphabet:
                raise ValueError("Alphabets of the sequences are not equal")
        if matrix is not None and not matrix.is_symmetric():
            raise ValueError(
                "A symmetric matrix is required for multiple sequence alignments"
            )

        if ProteinSequence.alphabet.extends(alphabet) and supports_protein:
            if matrix is not None and not supports_custom_protein_matrix:
                raise TypeError(
                    "The software does not support custom substitution "
                    "matrices for protein sequences"
                )
            return MSAInput("protein", sequences, matrix, sequences)

        if NucleotideSequence.alphabet_amb.extends(alphabet) and supports_nucleotide:
            if matrix is not None and not supports_custom_nucleotide_matrix:
                raise TypeError(
                    "The software does not support custom substitution "
                    "matrices for nucleotide sequences"
                )
            return MSAInput("nucleotide", sequences, matrix, sequences)

        # For all other sequence types, try to map onto protein sequences
        if not supports_protein:
            raise TypeError(
                f"The software cannot align sequences of type "
                f"{type(sequences[0]).__name__}: "
                f"No support for alignment of the mapped sequences"
            )
        if not supports_custom_protein_matrix:
            raise TypeError(
                f"The software cannot align sequences of type "
                f"{type(sequences[0]).__name__}: "
                f"No support for custom substitution matrices"
            )
        mapped_sequences: list[Sequence] = [
            _map_sequence(sequence) for sequence in sequences
        ]
        mapped_matrix = _map_matrix(matrix)
        return MSAInput("protein", mapped_sequences, mapped_matrix, sequences)

    def write_fasta(self, file: IO[str]) -> None:
        """
        Write the input sequences to a FASTA file, using their index as
        identifier.

        Parameters
        ----------
        file : file object
            The file the sequences are written to.
        """
        fasta_file = FastaFile()
        for i, sequence in enumerate(self.sequences):
            fasta_file[str(i)] = str(sequence)
        fasta_file.write(file)
        file.flush()

    def read_fasta(self, file: IO[str]) -> tuple[Alignment, np.ndarray]:
        """
        Read an MSA software's FASTA output into an :class:`Alignment`.

        Parameters
        ----------
        file : file object
            The FASTA file written by the MSA software.
            The sequences must be identified by their input index.

        Returns
        -------
        alignment : Alignment
            The multiple sequence alignment.
        order : ndarray, dtype=int
            The order of the sequences as intended by the MSA software.
        """
        alignment_file = FastaFile.read(file)
        seq_dict = OrderedDict(alignment_file)
        out_seq_str = [seq_dict[str(i)] for i in range(len(self.original_sequences))]
        trace = Alignment.trace_from_strings(out_seq_str)
        alignment = Alignment(list(self.original_sequences), trace, None)
        order = np.zeros(len(seq_dict), dtype=int)
        for i, seq_index in enumerate(seq_dict):
            order[i] = int(seq_index)
        return alignment, order


def resolve_gap_penalty(
    gap_penalty: float | tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    """
    Split a gap penalty into a gap opening and gap extension penalty.

    Parameters
    ----------
    gap_penalty : float | tuple[float, float] | None
        The gap penalty to split.
        If `None`, both gap opening and extension penalties are set to `None`.

    Returns
    -------
    gap_opening : float | None
        The gap opening penalty.
    gap_extension : float | None
        The gap extension penalty.
    """
    if gap_penalty is None:
        return None, None
    if isinstance(gap_penalty, SequenceABC):
        if gap_penalty[0] > 0 or gap_penalty[1] > 0:
            raise ValueError("Gap penalty must be negative")
        return gap_penalty[0], gap_penalty[1]
    elif isinstance(gap_penalty, numbers.Real):
        if gap_penalty > 0:
            raise ValueError("Gap penalty must be negative")
        return gap_penalty, gap_penalty
    else:
        raise TypeError("Gap penalty must be either float or tuple")


def _map_sequence(sequence: Sequence) -> ProteinSequence:
    """
    Map a sequence with an arbitrary alphabet into a
    :class:`ProteinSequence`, in order to support arbitrary sequence
    types in software that can handle protein sequences.

    Parameters
    ----------
    sequence : Sequence
        The sequence to be mapped.

    Returns
    -------
    mapped_sequence : ProteinSequence
        The mapped sequence.
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


def _map_matrix(matrix: SubstitutionMatrix | None) -> SubstitutionMatrix:
    """
    Map a :class:`SubstitutionMatrix` with an arbitrary alphabet into a
    :class:`SubstitutionMatrix` for protein sequences, in order to support
    arbitrary sequence types in software that can handle protein
    sequences.

    Parameters
    ----------
    matrix : SubstitutionMatrix
        The substitution matrix to be mapped.

    Returns
    -------
    mapped_matrix : SubstitutionMatrix
        The mapped substitution matrix.
    """
    if matrix is None:
        raise TypeError(
            "A substitution matrix must be provided for custom sequence types"
        )
    # Create a protein substitution matrix with the values taken
    # from the original matrix
    # All trailing symbols are filled with zeros
    old_length = len(matrix.get_alphabet1())
    new_length = len(ProteinSequence.alphabet)
    new_score_matrix = np.zeros((new_length, new_length), dtype=np.int32)
    new_score_matrix[:old_length, :old_length] = matrix.score_matrix()
    return SubstitutionMatrix(
        ProteinSequence.alphabet, ProteinSequence.alphabet, new_score_matrix
    )
