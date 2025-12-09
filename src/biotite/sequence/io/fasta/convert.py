# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.io.fasta"
__author__ = "Patrick Kunzmann"

import functools
import warnings
from collections import OrderedDict
import numpy as np
from biotite.sequence.align.alignment import Alignment, get_codes
from biotite.sequence.alphabet import AlphabetError, LetterAlphabet
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence

__all__ = [
    "get_sequence",
    "get_sequences",
    "set_sequence",
    "set_sequences",
    "get_alignment",
    "set_alignment",
    "get_a3m_alignments",
    "set_a3m_alignments",
]


def get_sequence(fasta_file, header=None, seq_type=None):
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
    seq_type : type[Sequence], optional
        The :class:`Sequence` subclass contained in the file.
        If not set, the type is automatically inferred as
        :class:`ProteinSequence` or :class:`NucleotideSequence`.
        For large sequence data it is recommended to set this parameter.

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
    return _convert_to_sequence(seq_str, seq_type)


def get_sequences(fasta_file, seq_type=None):
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
    seq_type : type[Sequence], optional
        The :class:`Sequence` subclass contained in the file.
        If not set, the type is automatically inferred as
        :class:`ProteinSequence` or :class:`NucleotideSequence`.
        For large sequence data it is recommended to set this parameter.

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
        seq_dict[header] = _convert_to_sequence(seq_str, seq_type)
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


def get_alignment(fasta_file, additional_gap_chars=("_",), seq_type=None):
    """
    Get an alignment from a :class:`FastaFile` instance.

    Parameters
    ----------
    fasta_file : FastaFile
        The :class:`FastaFile` to be accessed.
    additional_gap_chars : str, optional
        The characters to be treated as gaps.
    seq_type : type[Sequence], optional
        The :class:`Sequence` subclass contained in the file.
        If not set, the type is automatically inferred as
        :class:`ProteinSequence` or :class:`NucleotideSequence`.
        For large sequence data it is recommended to set this parameter.

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
    return Alignment.from_strings(
        seq_strings, functools.partial(_convert_to_sequence, seq_type=seq_type)
    )


def set_alignment(fasta_file, alignment, seq_names):
    """
    Fill a :class:`FastaFile` with gapped sequence strings from an alignment.

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


def get_a3m_alignments(a3m_file, seq_type=None):
    """
    Get pairwise sequence alignments from an *A3M*-formatted FASTA file.

    The *i*-th alignment is an alignment of the first sequence in the file (the query)
    to the *i+1*-th sequence in the file (the target).

    Parameters
    ----------
    a3m_file : FastaFile
        The A3M file to parse.
        The first sequence (the query) must not contain any deletions or insertions.
        All subsequent sequences indicate insertions and deletions by ``-`` or
        lower case characters, respectively.
    seq_type : type[Sequence], optional
        The :class:`Sequence` subclass contained in the file.
        If not set, the type is automatically inferred as
        :class:`ProteinSequence` or :class:`NucleotideSequence` from the query sequence.
        For large sequence data it is recommended to set this parameter.

    Returns
    -------
    alignments : list of Alignment
        Alignments of all sequences (excluding the query itself) to the query sequence.
        Each alignment is between the query (first element) and each target sequence
        (second element).
    """
    sequence_iterator = iter(a3m_file.values())
    query_str = next(sequence_iterator)
    query = _convert_to_sequence(query_str, seq_type)
    if isinstance(query, NucleotideSequence):
        factory = _convert_to_nucleotide
    elif isinstance(query, ProteinSequence):
        factory = _convert_to_protein
    else:
        factory = seq_type

    alignments = []
    for target_str in sequence_iterator:
        # The target sequence provides all information about the alignment
        # - matches/mismatches -> upper case
        # - gaps in query -> lower case
        # - gaps in target -> '-'
        target_byte_array = np.frombuffer(target_str.encode("ASCII"), dtype=np.ubyte)
        query_gaps = _is_lower(target_byte_array)
        target_gaps = _is_gap(target_byte_array)

        # Start with a trace filled with gaps (-1)
        trace = np.full((len(target_str), 2), -1, dtype=np.int64)
        # Fill the trace with the positions of the query sequence where there is no gap
        trace[~query_gaps, 0] = np.arange(len(query))
        # Do the same for the target sequence, but without the gap indicators
        # but remove the gap indicators from it first to get the actual sequence length
        trace[~target_gaps, 1] = np.arange(np.count_nonzero(~target_gaps))

        alignments.append(
            Alignment([query, factory(target_str.replace("-", ""))], trace)
        )

    return alignments


def set_a3m_alignments(a3m_file, alignments, query_label, target_labels):
    """
    Fill a :class:`FastaFile` with *A3M*-formatted alignments.

    Parameters
    ----------
    a3m_file : FastaFile
        The A3M file to fill.
    alignments : list of Alignment, length=n
        The pairwise alignments to fill the file with.
        The first sequence of each alignment must always be the same
        and will become the first sequence (the query) in the file.
    query_label : str
        The label for the query sequence.
    target_labels : iterable object of str, length=n
        The labels for the target sequences in the alignment.
    """
    query = alignments[0].sequences[0]
    a3m_file[query_label] = _convert_to_string(query, as_rna=False)

    for alignment, name in zip(alignments, target_labels, strict=True):
        if len(alignment.sequences) != 2:
            raise ValueError("Each alignment must be pairwise")
        if alignment.sequences[0] != query:
            raise ValueError(
                "The first sequence of each alignment must be the same as the query"
            )

        alignment = _as_global(alignment)

        code = get_codes(alignment)
        query_code = code[0]
        target_code = code[1]
        query_gaps = query_code == -1
        target_gaps = target_code == -1
        match_mask = ~query_gaps & ~target_gaps

        a3m_string_array = np.zeros(len(query_code), dtype="S1")
        # Indicate gaps in the target sequence with '-'
        a3m_string_array[target_gaps] = "-"
        # Keep gaps in the query sequence as lower case letters
        a3m_string_array[query_gaps] = np.char.lower(
            query.alphabet.decode_multiple(target_code[query_gaps], as_bytes=True)
        )
        # Matches/mismatches are indicated with upper case letters
        a3m_string_array[match_mask] = query.alphabet.decode_multiple(
            target_code[match_mask], as_bytes=True
        )
        a3m_file[name] = a3m_string_array.tobytes().decode("ASCII")


def _convert_to_sequence(seq_str, seq_type=None):
    # Set manually selected sequence type
    if seq_type is not None:
        # Do preprocessing as done without manual selection
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

    # Attempt to automatically determine sequence type

    try:
        return _convert_to_nucleotide(seq_str)
    except AlphabetError:
        pass
    try:
        prot_seq = _convert_to_protein(seq_str)
        # Raise Warning after conversion into 'ProteinSequence'
        # to wait for potential 'AlphabetError'
        if "U" in seq_str:
            warnings.warn(
                "ProteinSequence objects do not support selenocysteine (U), "
                "occurrences were substituted by cysteine (C)"
            )
        return prot_seq
    except AlphabetError:
        raise ValueError(
            "FASTA data cannot be converted either to "
            "'NucleotideSequence' nor to 'ProteinSequence'"
        )


def _convert_to_protein(seq_str):
    """
    Replace selenocysteine with cysteine and pyrrolysine with lysine.
    """
    return ProteinSequence(seq_str.upper().replace("U", "C").replace("O", "K"))


def _convert_to_nucleotide(seq_str):
    """
    For nucleotides uracil is represented by thymine and there is only
    one letter for completely unknown nucleotides
    """
    return NucleotideSequence(seq_str.upper().replace("U", "T").replace("X", "N"))


def _convert_to_string(sequence, as_rna):
    if not isinstance(sequence.get_alphabet(), LetterAlphabet):
        raise ValueError(
            "Only sequences using single letter alphabets can be stored in a FASTA file"
        )
    if isinstance(sequence, NucleotideSequence) and as_rna:
        return str(sequence).replace("T", "U")
    else:
        return str(sequence)


def _as_global(alignment):
    """
    Convert a semi-global alignment into a global alignment.

    A semi-global alignment is an alignment, where alignment columns for terminal
    gaps are not included.
    """
    trace = alignment.trace
    sequence_lengths = np.array([len(sequence) for sequence in alignment.sequences])

    start_positions = []
    end_positions = []
    for i in range(trace.shape[1]):
        trace_for_seq = trace[:, i]
        trace_wo_gaps = trace_for_seq[trace_for_seq != -1]
        start_positions.append(trace_wo_gaps[0])
        end_positions.append(trace_wo_gaps[-1])
    start_positions = np.array(start_positions)
    end_positions = np.array(end_positions)
    if (
        np.count_nonzero(start_positions != 0) > 1
        or np.count_nonzero(end_positions != sequence_lengths - 1) > 1
    ):
        # If multiple sequences do not run from beginning to end,
        # the alignment is not semi-global, but local
        raise ValueError("Alignment is local, but a semi-global alignment is required")

    trace_parts = [trace]
    if not (start_positions == 0).all():
        # We need to add a prefix to the alignment, which has gaps for all sequences
        # except for one
        seq_index_with_missing_start = np.where(start_positions != 0)[0][0]
        trace_prefix = np.full(
            (start_positions[seq_index_with_missing_start], trace.shape[1]),
            -1,
            dtype=int,
        )
        trace_prefix[:, seq_index_with_missing_start] = np.arange(len(trace_prefix))
        trace_parts.insert(0, trace_prefix)
    if not (end_positions == sequence_lengths).all():
        # The same needs to be done for the end of the alignment
        seq_index_with_missing_end = np.where(end_positions != sequence_lengths)[0][0]
        end_position = end_positions[seq_index_with_missing_end]
        seq_length = sequence_lengths[seq_index_with_missing_end]
        trace_suffix = np.full(
            (seq_length - end_position - 1, trace.shape[1]), -1, dtype=int
        )
        trace_suffix[:, seq_index_with_missing_end] = np.arange(
            end_position + 1, end_position + 1 + len(trace_suffix)
        )
        trace_parts.append(trace_suffix)

    trace = np.concatenate(trace_parts, axis=0)
    return Alignment(alignment.sequences, trace)


def _is_lower(characters):
    return (characters >= ord("a")) & (characters <= ord("z"))


def _is_gap(characters):
    return characters == ord("-")
