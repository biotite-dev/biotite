# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.clustal"
__author__ = "Biotite contributors"

import functools
import warnings
from collections import OrderedDict
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.alphabet import AlphabetError, LetterAlphabet
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence

__all__ = ["get_alignment", "set_alignment"]


def get_alignment(clustal_file, seq_type=None):
    """
    Get an alignment from a :class:`ClustalFile` instance.

    Parameters
    ----------
    clustal_file : ClustalFile
        The :class:`ClustalFile` to be accessed.
    seq_type : type[Sequence], optional
        The :class:`Sequence` subclass contained in the file.
        If not set, the type is automatically inferred as
        :class:`ProteinSequence` or :class:`NucleotideSequence`.
        For large sequence data it is recommended to set this parameter.

    Returns
    -------
    alignment : Alignment
        The alignment from the :class:`ClustalFile`.
    """
    seq_strings = list(clustal_file.values())
    if len(seq_strings) < 2:
        raise ValueError(
            f"ClustalW file contains {len(seq_strings)} sequence(s), "
            f"but an alignment requires at least 2"
        )
    return Alignment.from_strings(
        seq_strings,
        functools.partial(_convert_to_sequence, seq_type=seq_type),
    )


def set_alignment(clustal_file, alignment, seq_names, line_length=60):
    """
    Fill a :class:`ClustalFile` with gapped sequence strings from an
    alignment.

    Parameters
    ----------
    clustal_file : ClustalFile
        The :class:`ClustalFile` to be accessed.
    alignment : Alignment
        The alignment to be set.
    seq_names : iterable object of str
        The names for the sequences in the alignment.
        Must have the same length as the sequence count in `alignment`.
    line_length : int, optional
        The number of sequence characters per line in each block.
        Default is 60.
    """
    if line_length < 1:
        raise ValueError(
            f"'line_length' must be at least 1, got {line_length}"
        )
    gapped_seq_strings = alignment.get_gapped_sequences()
    if len(gapped_seq_strings) != len(seq_names):
        raise ValueError(
            f"Alignment has {len(gapped_seq_strings)} sequences, "
            f"but {len(seq_names)} names were given"
        )
    # Clear any existing entries before writing new data
    clustal_file._entries.clear()
    for name, seq_str in zip(seq_names, gapped_seq_strings):
        clustal_file._entries[name] = seq_str
    # Rebuild the text lines with the specified line length
    clustal_file._rebuild_lines(line_length=line_length)


def _convert_to_sequence(seq_str, seq_type=None):
    if seq_type is not None:
        if seq_type == NucleotideSequence:
            return _convert_to_nucleotide(seq_str)
        elif seq_type == ProteinSequence:
            if "U" in seq_str:
                warnings.warn(
                    "ProteinSequence objects do not support selenocysteine "
                    "(U), occurrences were substituted by cysteine (C)"
                )
            return _convert_to_protein(seq_str)
        else:
            return seq_type(seq_str)

    try:
        return _convert_to_nucleotide(seq_str)
    except AlphabetError:
        pass
    try:
        prot_seq = _convert_to_protein(seq_str)
        if "U" in seq_str:
            warnings.warn(
                "ProteinSequence objects do not support selenocysteine (U), "
                "occurrences were substituted by cysteine (C)"
            )
        return prot_seq
    except AlphabetError:
        raise ValueError(
            "ClustalW data cannot be converted either to "
            "'NucleotideSequence' nor to 'ProteinSequence'"
        )


def _convert_to_protein(seq_str):
    return ProteinSequence(seq_str.upper().replace("U", "C").replace("O", "K"))


def _convert_to_nucleotide(seq_str):
    return NucleotideSequence(seq_str.upper().replace("U", "T").replace("X", "N"))
