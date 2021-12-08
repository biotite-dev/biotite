# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.fastq"
__author__ = "Patrick Kunzmann"

from collections import OrderedDict
from ...sequence import Sequence
from ...alphabet import AlphabetError, LetterAlphabet
from ...seqtypes import NucleotideSequence
from ...align.alignment import Alignment

__all__ = ["get_sequence", "get_sequences", "set_sequence", "set_sequences"]


def get_sequence(fastq_file, header=None):
    """
    Get a sequence and quality scores from a `FastqFile` instance.
    
    Parameters
    ----------
    fastq_file : FastqFile
        The `FastqFile` to be accessed.
    header : str, optional
        The identifier to get the sequence and scores from.
        By default, the first sequence of the file is returned.
    
    Returns
    -------
    sequence : NucleotideSequence
        The requested sequence.
    scores : ndarray, dtype=int
        The requested scores.
    """
    if header is not None:
        seq_str, scores = fastq_file[header]
    else:
        # Return first (and probably only) sequence of file
        seq_str = None
        scores = None
        for seq_str, scores in fastq_file.values():
            break
        if seq_str is None:
            raise ValueError("File does not contain any sequences")
    processed_seq_str = seq_str.replace("U","T").replace("X","N")
    return NucleotideSequence(processed_seq_str), scores


def get_sequences(fastq_file):
    """
    Get a dictionary from a `FastqFile` instance,
    where identifiers are keys and sequence-score-tuples are values.
    
    Parameters
    ----------
    fastq_file : FastqFile
        The `Fastqile` to be accessed.
    
    Returns
    -------
    seq_dict : dict
        A dictionary containing identifiers as keys and
        (`NucleotideSequence`, `ndarray`) tuples as values.
    """
    seq_dict = OrderedDict()
    for header, (seq_str, scores) in fastq_file.items():
        processed_seq_str = seq_str.replace("U","T").replace("X","N")
        seq_dict[header] = NucleotideSequence(processed_seq_str), scores
    return seq_dict


def set_sequence(fastq_file, sequence, scores, header=None, as_rna=False):
    """
    Set a sequence and a quality score array in a `FastqFile` instance.
    
    Parameters
    ----------
    fastq_file : FastqFile
        The `FastqFile` to be accessed.
    sequence : NucleotideSequence
        The sequence to be set.
    scores : ndarray, dtype=int
        The quality scores to be set.
    header : str, optional
        The identifier for the sequence. Default is 'sequence'.
    as_rna : bool, optional
        If set to true, the sequence symbol ``'T'`` will be replaced
        by ``'U'``.
    """
    if header is None:
        header = "sequence"
    fastq_file[header] = _convert_to_string(sequence, as_rna), scores


def set_sequences(fastq_file, sequence_dict, as_rna=False):
    """
    Set sequences in a `FastqFile` instance from a dictionary.
    
    Parameters
    ----------
    fastq_file : FastqFile
        The `FastqFile` to be accessed.
    sequence_dict : dict
        A dictionary containing the sequences and scores to be set.
        Identifiers are keys,
        (`NucleotideSequence`, `ndarray`) tuples are values.
    as_rna : bool, optional
        If set to true, the sequence symbol ``'T'`` will be replaced
        by ``'U'``.
    """
    for header, (sequence, scores) in sequence_dict.items():
        fastq_file[header] = _convert_to_string(sequence, as_rna), scores


def _convert_to_string(sequence, as_rna):
    if as_rna:
        return(str(sequence).replace("T", "U"))
    else:
        return(str(sequence))