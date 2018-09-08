# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import numpy as np
import copy
import textwrap
from ..alphabet import LetterAlphabet


__all__ = ["Alignment", "get_codes", "get_symbols", "get_sequence_identity",
           "score"]


class Alignment(object):
    """
    An `Alignment` object stores information about which symbols of
    *n* sequences are aligned to each other and it stores the
    corresponding alignment score.
    
    Instead of saving a list of aligned symbols, this class saves the
    original *n* sequences, that were aligned, and a so called *trace*,
    which indicate the aligned symbols of these sequences.
    The trace is a *(m x n)* `ndarray` with alignment length *m* and
    sequence count *n*.
    Each element of the trace is the index in the corresponding
    sequence. A gap is represented by the value -1.
    
    Furthermore this class provides multiple utility functions for
    conversion into strings in order to make the alignment human
    readable.
    
    Unless an `Alignment` object is the result of an multiple sequence
    alignment, the object will contain only two sequences.
    
    All attributes of this class are publicly accessible.
    
    Parameters
    ----------
    sequences : list
        A list of aligned sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int, optional
        Alignment score.
    
    Attributes
    ----------
    sequences : list
        A list of aligned sequences.
    trace : ndarray, dtype=int, shape=(n,m)
        The alignment trace.
    score : int
        Alignment score.
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("CGTCAT")
    >>> seq2 = NucleotideSequence("TCATGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTCAT--
    --TCATGC
    >>> print(ali.trace)
    [[ 0 -1]
     [ 1 -1]
     [ 2  0]
     [ 3  1]
     [ 4  2]
     [ 5  3]
     [-1  4]
     [-1  5]]
    >>> print(ali[1:4].trace)
     [ 1 -1]
     [ 2  0]
     [ 3  1]]
    >>> print(ali[1:4,0].trace)
    [1 2 3]
    """
    
    
    def __init__(self, sequences, trace, score=None):
        self.sequences = sequences.copy()
        self.trace = trace
        self.score = score
    
    def _gapped_str(self, seq_index):
        seq_str = ""
        for i in range(len(self.trace)):
            j = self.trace[i][seq_index]
            if j != -1:
                seq_str += self.sequences[seq_index][j]
            else:
                seq_str += "-"
        return seq_str
    
    def get_gapped_sequences(self):
        """
        Get a the string representation of the gapped sequences.
        
        Returns
        -------
        sequences : list of str
            The list of gapped sequence strings. The order is the same
            as in `Alignment.sequences`.
        """
        return [self._gapped_str(i) for i in range(len(self.sequences))]
    
    def __str__(self):
        # Check if any of the sequences
        # has an non-single letter alphabet
        all_single_letter = True
        for seq in self.sequences:
            if not isinstance(seq.get_alphabet(), LetterAlphabet):
                all_single_letter = False
        if all_single_letter:
            # First dimension: sequence number,
            # second dimension: line number
            seq_str_lines_list = []
            wrapper = textwrap.TextWrapper(break_on_hyphens=False)
            for i in range(len(self.sequences)):
                seq_str_lines_list.append(wrapper.wrap(self._gapped_str(i)))
            ali_str = ""
            for row_i in range(len(seq_str_lines_list[0])):
                for seq_j in range(len(seq_str_lines_list)):
                    ali_str += seq_str_lines_list[seq_j][row_i] + "\n"
                ali_str += "\n"
            # Remove final line breaks
            return ali_str[:-2]
        else:
            super().__str__()
    
    def __getitem__(self, index):
        if isinstance(index, tuple):
            new_trace = self.trace[index]
            if isinstance(index[1], (list, tuple, np.ndarray)):
                new_sequences = [self.sequences[i] for i in index[1]]
            else:
                new_sequences = self.sequences[index[1]]
            return Alignment(new_sequences, new_trace, self.score)
        elif isinstance(index, slice):
            return Alignment(self.sequences[:], self.trace[index], self.score)
        else:
            raise IndexError(f"Invalid alignment index type "
                             f"'{type(index).__name__}'")
    
    def __iter__(self):
        raise TypeError("'Alignment' object is not iterable")
    
    @staticmethod
    def trace_from_strings(seq_str_list):
        """
        Create a trace from strings that represent aligned sequences.
        
        Parameters
        ----------
        seq_str_list : list of str
            The strings, where each each one represents a sequence
            in an alignment.
        
        Returns
        -------
        trace : ndarray, dtype=int, shape=(n,2)
            The created trace.
        """
        seq_i = np.zeros(len(seq_str_list))
        trace = np.full(( len(seq_str_list[0]), len(seq_str_list) ),
                        -1, dtype=int)
        # Get length of string (same length for all strings)
        # rather than length of list
        for pos_i in range(len(seq_str_list[0])):
            for str_j in range(len(seq_str_list)):
                if seq_str_list[str_j][pos_i] == "-":
                    trace[pos_i, str_j] = -1
                else:
                    trace[pos_i, str_j] = seq_i[str_j]
                    seq_i[str_j] += 1
        return trace


def get_codes(alignment):
    """
    Get the sequence codes for the alignment.

    The codes are built from the trace:
    Instead of the indices of the aligned symbols, the return value
    contains the corresponding symbol codes for each index.
    Gaps are still represented by *-1*.
    
    Parameters
    ----------
    alignment : Alignment
        The alignment to get the sequence codes for.
    
    Returns
    -------
    codes : ndarray, dtype=int, shape=(n,m)
        The sequence codes for the alignment.
        The shape is *(n,m)* for *n* sequences and *m* alignment cloumn.
        The array uses *-1* values for gaps.
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("CGTCAT")
    >>> seq2 = NucleotideSequence("TCATGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTCAT--
    --TCATGC
    >>> print(align.get_codes(ali))
    [[ 1  2  3  1  0  3 -1 -1]
     [-1 -1  3  1  0  3  2  1]]
    """
    trace = alignment.trace
    # -1 is code value for gaps
    codes = np.full(trace.shape, -1, dtype=int)
    for i in range(trace.shape[0]):
        for j in range(trace.shape[1]):
            # Get symbol code for each index in trace
            index = trace[i,j]
            # Code is set to -1 for gaps
            if index != -1:
                codes[i,j] = alignment.sequences[j].code[index]
    # Transpose to have the number of sequences as first dimension
    return codes.transpose()


def get_symbols(alignment):
    """
    Similar to `get_codes()`, but contains the decoded symbols instead
    of codes.
    Gaps are still represented by *None* values.
    
    Parameters
    ----------
    alignment : Alignment
        The alignment to get the symbols for.
    
    Returns
    -------
    symbols : list of list
        The nested list of symbols.
    
    See Also
    --------
    get_codes
    """
    codes = get_codes(alignment)
    symbols = [None] * codes.shape[0]
    for i in range(codes.shape[0]):
        symbols_for_seq = [None] * codes.shape[1]
        alphabet = alignment.sequences[i].get_alphabet()
        for j in range(codes.shape[1]):
            code = codes[i,j]
            # Store symbol in the list
            # For gaps (-1) this condition fails
            # and the list keeps a 'None'
            if code != -1:
                symbols_for_seq[j] = alphabet.decode(code)
        symbols[i] = symbols_for_seq
    return symbols


def get_sequence_identity(alignment, mode="not_terminal"):
    """
    Calculate the sequence identity for an alignment.

    The identity is equal to the matches divided by a measure for the
    length of the alignment that depends on the `mode` parameter.
    
    Parameters
    ----------
    alignment : Alignment
        The alignment to calculate the identity for.
    mode : {'all', 'not_terminal', 'shortest'}, optional
        The calculation mode for alignment length.

            - **all** - Count all alignment columns.
            - **not_terminal** - Count all alignment columns of
              non-terminal gaps.
            - **shortest** - Use the shortest sequence.

        Default is *not_terminal*.
    
    Returns
    -------
    identity : float
        The sequence identity, ranging between 0 and 1.
    """
    if mode not in ["all", "not_terminal", "shortest"]:
        raise ValueError(f"'{mode}' is an invalid calculation mode")
    trace = alignment.trace
    codes = get_codes(alignment)
    
    if mode in ["not_terminal", "shortest"]:
        # Ignore terminal gaps
        # -> get start and exclusive stop column of the trace
        # excluding terminal gaps
        start_index = -1
        for i in range(len(trace)):
            # Check if all sequences have no gap at the given position
            if (trace[i] != -1).all():
                start_index = i
                break
        # Reverse iteration
        stop_index = -1
        for i in range(len(trace)-1, -1, -1):
            # Check if all sequences have no gap at the given position
            if (trace[i] != -1).all():
                stop_index = i+1
                break
        if start_index == -1 or stop_index == -1:
            raise ValueError("Alignment entirely consists of terminal gaps")
    else:
        # 'all' -> count all columns, entire trace
        start_index = 0
        stop_index = len(trace)
    
    # Count matches
    matches = 0
    for i in range(start_index, stop_index):
        column = codes[:,i]
        # One unique value -> all symbols match
        if len(np.unique(column)) == 1:
            matches += 1
    
    #Calculate iddentity
    if mode == "shortest":
        shortest_length = min([len(seq) for seq in alignment.sequences])
        return matches / shortest_length
    else:
        return matches / (stop_index - start_index)


def score(alignment, matrix, gap_penalty=-10, terminal_penalty=True):
    """
    Calculate the similarity score of an alignment.

    If the alignment contains more than two sequences,
    all pairwise scores are counted.
    
    Parameters
    ----------
    alignment : Alignment
        The alignment to calculate the identity for.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_penalty : int or (tuple, dtype=int), optional
        If an integer is provided, the value will be interpreted as
        general gap penalty. If a tuple is provided, an affine gap
        penalty is used. The first integer in the tuple is the gap
        opening penalty, the second integer is the gap extension
        penalty.
        The values need to be negative. (Default: *-10*)
    terminal_penalty : bool, optional
        If true, gap penalties are applied to terminal gaps.
        (Default: True)
    
    Returns
    -------
    score : int
        The similarity score.
    """
    codes = get_codes(alignment)
    matrix = matrix.score_matrix()

    # Sum similarity scores (without gaps)
    score = 0
    # Iterate over all positions
    for pos in range(codes.shape[1]):
        column = codes[:, pos]
        # Iterate over all possible pairs
        # Do not count self-similarity
        # and do not count similarity twice (not S(i,j) and S(j,i))
        for i in range(codes.shape[0]):
            for j in range(i+1, codes.shape[0]):
                code_i = column[i]
                code_j = column[j]
                # Ignore gaps
                if code_i != -1 and code_j != -1:
                    score += matrix[code_i, code_j]
    # Sum gap penalties
    if type(gap_penalty) == int:
        gap_open = gap_penalty
        gap_ext = gap_penalty
    elif type(gap_penalty) == tuple:
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    # Iterate over all sequences
    for seq_code in codes:
        in_gap = False
        if terminal_penalty:
            start_index = 0
            stop_index = len(seq_code)
        else:
            # Find a start and stop index excluding terminal gaps
            start_index, stop_index = _identify_terminal_gaps(seq_code)
        for i in range(start_index, stop_index):
            if seq_code[i] == -1:
                if in_gap:
                    score += gap_ext
                else:
                    score += gap_open
                in_gap = True
            else:
                in_gap = False
    return score


def _identify_terminal_gaps(seq_code):
    # Find a start and stop index excluding terminal gaps
    start_index = -1
    for i in range(len(seq_code)):
        # Check if the sequence has a gap at the given position
        if seq_code[i] != -1:
            start_index = i
            break
    # Reverse iteration
    stop_index = -1
    for i in range(len(seq_code)-1, -1, -1):
        if (seq_code[i] != -1).all():
            stop_index = i+1
            break
    if start_index == -1 or stop_index == -1:
        raise ValueError("Sequence is empty")
    return start_index, stop_index