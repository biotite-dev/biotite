# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ...sequence import Sequence
from ...alphabet import AlphabetError
from ...seqtypes import NucleotideSequence, ProteinSequence

__all__ = ["get_sequence", "get_sequences", "set_sequence", "set_sequences"]


def get_sequence(fasta_file, header=None):
    """
    Get a sequence from a `FastaFile` instance.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The `FastaFile` to be accessed.
    header : str, optional
        The header to get the sequence from. By default, the first
        sequence of the file is returned.
    
    Raises
    ------
    ValueError
        If the sequence data can be neither converted into a
        `NucleotideSequence` nor a `ProteinSequence`.
    
    Returns
    -------
    sequence : `NucleotideSequence` or `ProteinSequence`
        `NucleotideSequence` if the sequence string fits the
        corresponding alphabet, `ProteinSequence` otherwise.
    """
    if header is not None:
        seq_str = fasta_file[header]
    else:
        # Return first (and probably only) sequence of file
        for header, seq_str in fasta_file:
            break
    # Determine the sequence type:
    # If NucleotideSequence can be created it is a DNA sequence,
    # otherwise protein sequence
    return _convert_to_sequence(seq_str)


def get_sequences(fasta_file):
    """
    Get dictionary from a `FastaFile` instance,
    where headers are keys and sequences are values.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The `FastaFile` to be accessed.
    
    Raises
    ------
    ValueError
        If at least on of the sequence strings can be neither converted
        into a `NucleotideSequence` nor a `ProteinSequence`.
    
    Returns
    -------
    seq_dict : dict
        A dictionary containg `NucleotideSequence` and/or
        `ProteinSequence` instances.
    """
    seq_str_dict = dict(fasta_file)
    seq_dict = {}
    for header, seq_str in seq_str_dict.items():
        seq_dict[header] = _convert_to_sequence(seq_str)
    return seq_dict


def set_sequence(fasta_file, sequence, header=None):
    """
    Set a sequence in a `FastaFile` instance.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The `FastaFile` to be accessed.
    sequence : Sequence
        The sequence to be set.
    header : str, optional
        The header for the sequence. Default is 'sequence'.
    
    Raises
    ------
    ValueError
        If the sequence's alphabet uses symbols other than single
        letters.
    """
    if header is None:
        header = "sequence"
    fasta_file[header] = _convert_to_string(sequence)


def set_sequences(fasta_file, sequence_dict):
    """
    Set sequences in a `FastaFile` instance from a dictionary.
    
    Parameters
    ----------
    fasta_file : FastaFile
        The `FastaFile` to be accessed.
    sequence_dict : dict
        A dictionary containing the sequences to be set.
        Header are keys, `Sequence` instances are values. 
    
    Raises
    ------
    ValueError
        If the sequences alphabets uses symbols other than single
        letters.
    """
    seq_str_dict = {}
    for header, sequence in sequence_dict.items():
        fasta_file[header] = _convert_to_string(sequence)


def _convert_to_sequence(seq_str):
    try:
        return NucleotideSequence(seq_str)
    except AlphabetError:
        pass
    try:
        return ProteinSequence(seq_str)
    except AlphabetError:
        raise ValueError("FASTA data cannot be converted either to "
                         "NucleotideSequence nor to Protein Sequence")


def _convert_to_string(sequence):
    if not sequence.alphabet.is_letter_alphabet():
        raise ValueError("Only sequences using single letter alphabets"
                         "can be stored in a FASTA file")
    return(str(sequence))