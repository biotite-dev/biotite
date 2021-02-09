# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.fasta"
__author__ = "Patrick Kunzmann"

import warnings
from collections import OrderedDict
from ...sequence import Sequence
from ...alphabet import AlphabetError, LetterAlphabet
from ...seqtypes import NucleotideSequence, ProteinSequence
from ...align.alignment import Alignment

__all__ = ["get_sequence", "get_sequences", "set_sequence", "set_sequences",
           "get_alignment", "set_alignment"]


def get_sequence(fasta_file, header=None):
    """
    Get a sequence from a :class:`FastaFile` instance.

    The type of sequence is guessed from the sequence string:
    First, a conversion into a :class:`NucleotideSequence` and
    second a conversion into a :class:`ProteinSequence` is tried.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    header : str, optional
        The header to get the sequence from. By default, the first
        sequence of the file is returned.
    
    Returns
    -------
    sequence : NucleotideSequence or ProteinSequence
        The requested sequence in the `FastaFile`.
        :class:`NucleotideSequence` if the sequence string fits the
        corresponding alphabet, :class:`ProteinSequence` otherwise.
    
    Raises
    ------
    ValueError
        If the sequence data can be neither converted into a
        :class:`NucleotideSequence` nor a :class:`ProteinSequence`.
    """
    if header is not None:
        seq_str = fasta_file[header]
    else:
        # Return first (and probably only) sequence of file
        seq_str = None
        for seq_str in fasta_file.values():
            break
        if seq_str is None:
            raise ValueError("File does not contain any sequences")
    # Determine the sequence type:
    # If NucleotideSequence can be created it is a DNA sequence,
    # otherwise protein sequence
    return _convert_to_sequence(seq_str)


def get_sequences(fasta_file):
    """
    Get dictionary from a :class:`FastaFile` instance,
    where headers are keys and sequences are values.

    The type of sequence is guessed from the sequence string:
    First, a conversion into a :class:`NucleotideSequence` and
    second a conversion into a :class:`ProteinSequence` is tried.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    
    Returns
    -------
    seq_dict : dict
        A dictionary that maps headers to
        :class:`NucleotideSequence` and/or :class:`ProteinSequence`
        instances as values.
    
    Raises
    ------
    ValueError
        If at least on of the sequence strings can be neither converted
        into a :class:`NucleotideSequence` nor a
        :class:`ProteinSequence`.
    """
    seq_dict = OrderedDict()
    for header, seq_str in fasta_file.items():
        seq_dict[header] = _convert_to_sequence(seq_str)
    return seq_dict


def set_sequence(fasta_file, sequence, header=None, as_rna=False):
    """
    Set a sequence in a :class:`FastaFile` instance.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    sequence : Sequence
        The sequence to be set.
    header : str, optional
        The header for the sequence. Default is ``'sequence'``.
    as_rna : bool, optional
        If set to true, ``'T'`` will be replaced by ``'U'``,
        if a :class:`NucleotideSequence` was given.
    
    Raises
    ------
    ValueError
        If the sequence's alphabet uses symbols other than single
        characters.
    """
    if header is None:
        header = "sequence"
    fasta_file[header] = _convert_to_string(sequence, as_rna)


def set_sequences(fasta_file, sequence_dict, as_rna=False):
    """
    Set sequences in a :class:`FastaFile` instance from a dictionary.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    sequence_dict : dict
        A dictionary containing the sequences to be set.
        Header are keys, :class:`Sequence` instances are values.
    as_rna : bool, optional
        If set to true, ``'T'`` will be replaced by ``'U'``,
        if a :class:`NucleotideSequence` was given.
    
    Raises
    ------
    ValueError
        If the sequences alphabets uses symbols other than single
        characters.
    """
    for header, sequence in sequence_dict.items():
        fasta_file[header] = _convert_to_string(sequence, as_rna)


def get_alignment(fasta_file, additional_gap_chars=("_",)):
    """
    Get an alignment from a :class:`FastaFile` instance.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    additional_gap_chars : str, optional
        The characters to be treated as gaps.
    
    Returns
    -------
    alignment : Alignment
        The alignment from the :class:`FastaFile`.
    """
    seq_strings = list(fasta_file.values())
    # Replace additional gap symbols with default gap symbol ('-')
    for char in additional_gap_chars:
        for i, seq_str in enumerate(seq_strings):
            seq_strings[i] = seq_str.replace(char, "-")
    # Remove gaps for creation of sequences
    sequences = [_convert_to_sequence(seq_str.replace("-",""))
                 for seq_str in seq_strings]
    trace = Alignment.trace_from_strings(seq_strings)
    return Alignment(sequences, trace, score=None)


def set_alignment(fasta_file, alignment, seq_names):
    """
    Fill a :class:`FastaFile` with gapped sequence strings from an
    alignment.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    alignment : Alignment
        The alignment to be set. 
    seq_names : iterable object of str
        The names for the sequences in the alignment.
        Must have the same length as the sequence count in `alignment`.
    """
    gapped_seq_strings = alignment.get_gapped_sequences()
    if len(gapped_seq_strings) != len(seq_names):
        raise ValueError(
            f"Alignment has {len(gapped_seq_strings)} sequences, "
            f"but {len(seq_names)} names were given"
        )
    for i in range(len(gapped_seq_strings)):
        fasta_file[seq_names[i]] = gapped_seq_strings[i]


def _convert_to_sequence(seq_str):
    # Biotite alphabets for nucleotide and proteins
    # do not accept lower case letters
    seq_str = seq_str.upper()
    try:
        # For nucleotides uracil is represented by thymine and
        # there is is only one letter for completely unknown nucleotides
        return NucleotideSequence(seq_str.replace("U","T").replace("X","N"))
    except AlphabetError:
        pass
    try:
        if "U" in seq_str:
            warn = True
            seq_str = seq_str.replace("U", "C")
        else:
            warn = False
        prot_seq = ProteinSequence(seq_str)
        # Raise Warning after conversion into 'ProteinSequence'
        # to wait for potential 'AlphabetError'
        if warn:
            warnings.warn(
                "ProteinSequence objects do not support selenocysteine (U), "
                "occurrences were substituted by cysteine (C)"
            )
        return prot_seq
    except AlphabetError:
        raise ValueError("FASTA data cannot be converted either to "
                         "'NucleotideSequence' nor to 'ProteinSequence'")


def _convert_to_string(sequence, as_rna):
    if not isinstance(sequence.get_alphabet(), LetterAlphabet):
        raise ValueError("Only sequences using single letter alphabets "
                         "can be stored in a FASTA file")
    if isinstance(sequence, NucleotideSequence) and as_rna:
        return(str(sequence).replace("T", "U"))
    else:
        return(str(sequence))
