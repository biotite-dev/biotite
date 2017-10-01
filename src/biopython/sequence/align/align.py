# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from .matrix import SubstitutionMatrix
from ..sequence import Sequence
import numpy as np
import copy
import textwrap
from ...extensions import has_c_extensions, uses_c_extensions
if has_c_extensions():
    from .calign import c_fill_align_table, c_follow_trace
    

__all__ = ["Alignment", "simple_score", "align_global", "align_local"]



class Alignment(object):
    """
    An `Alignment` object stores information about which symbols of
    *n* sequences are aligned to each other and it stores the
    corresponding alignment score.
    
    Rather than saving a list of aligned symbols, this class saves the
    original *n* sequences, that were aligned, and a so called *trace*,
    which indicate the aligned symbols of these sequences.
    The trace is a *(m x n)* `ndarray` with alignment length *m*, where
    the first dimension is the position in the alignment and the second
    dimension represents the sequence.
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
    score : int
        Alignment score.
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("CGTT")
    >>> seq2 = NucleotideSequence("TTGC")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_global(seq1, seq2, matrix)[0]
    >>> print(ali)
    CGTT--
    --TTGC
    >>> print(ali.trace)
    [[ 0 -1]
     [ 1 -1]
     [ 2  0]
     [ 3  1]
     [-1  2]
     [-1  3]]
    >>> print(ali[1:4].trace)
    [[ 1 -1]
     [ 2  0]
     [ 3  1]]
    >>> print(ali[1:4,0].trace)
    [1 2 3]
    """
    
    
    def __init__(self, sequences, trace, score):
        self.sequences = copy.copy(sequences)
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
    
    def __str__(self):
        # Check if any of the sequences
        # has an non-single letter alphabet
        all_single_letter = True
        for seq in self.sequences:
            if not seq.get_alphabet().is_letter_alphabet():
                all_single_letter = False
        if all_single_letter:
            seq_str_lines_list = []
            for i in range(len(self.sequences)):
                seq_str_lines_list.append(textwrap.wrap(self._gapped_str(i)))
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
            return Alignment(self.sequences[index[1]], self.trace[index],
                             self.score)
        elif isinstance(index, slice):
            return Alignment(self.sequences[:], self.trace[index], self.score)
        else:
            raise IndexError("Invalid alignment index")
    
    @staticmethod
    def trace_from_strings(seq_str_list):
        """
        Create a trace from strings that represent aligned sequences.
        
        Parameters
        ----------
        seq_str_list : list
            The strings, where each each one represents a sequence
            in an alignment.
        
        Returns
        -------
        trace : ndarray, dtype=int, shype=(n,2)
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


def simple_score(seq1, seq2, matrix):
    """
    Score the similarity of two sequences.
    
    This is equal to a global alignment without introducing gaps.
    Therefore both sequences need to have the same length.
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences, whose similarity should be scored.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    
    Returns
    -------
    score : int
        The resulting score.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequence lengths of {:d} and {:d} are not equal"
                         .format( len(seq1), len(seq2) ))
    if (matrix.get_alphabet1() != seq1.get_alphabet() and
        matrix.get_alphabet2() != seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    seq1_code = seq1.code
    seq2_code = seq2.code
    score = 0
    for i in range(len(seq1)):
        score += matrix.get_score_by_code(seq1_code[i], seq2_code[i])
    return score


def align_global(seq1, seq2, matrix, gap_opening=-3, gap_extension=-1):
    """
    Perform a global alignment based on the Needleman-Wunsch
    algorithm [1]_.
    
    This algorithm yields an optimal alignment, i.e. the sequences
    are aligned in the way that results in the highest similarity score.
    This operation can be very time and space consuming, because both
    scale linearly with each sequence length.
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_opening : int, optional
        Negative value which is used as penalty for gap openings.
    gap_extension : int, optional
        Negative value which is used as penalty for gap extensions.
    
    Returns
    -------
    alignments : list, type=Alignment
        A list of alignments. Each alignment in the list has
        the same maximum similarity score.
    
    References
    ----------
    
    .. [1] SB Needleman, CD Wunsch,
       "A general method applicable to the search for similarities
       in the amino acid sequence of two proteins."
       J Mol Biol, 48, 443-453 (1970).
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("ATACGCTTGCT")
    >>> seq2 = NucleotideSequence("AGGCGCAGCT")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_global(seq1, seq2, matrix,
    ...                    gap_opening=-6, gap_extension=-2)
    >>> for a in ali:
    ...     print(a, "\\n")
    ATACGCTTGCT
    AGGCGCA-GCT 
    <BLANKLINE>
    ATACGCTTGCT
    AGGCGC-AGCT 
    <BLANKLINE>
    """
    # Check matrix alphabets
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("Substitution matrix is inappropriate "
                             "for the given sequences")
    # This implementation uses transposed tables in comparison
    # to the original algorithm
    # Therefore the first sequence is one the left
    # and the second sequence is at the top  
    # The table for saving the scores
    score_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
    # The table the directions a field came from
    # A "1" in the corresponding bit means
    # the field came from this direction
    # The values: bit 1 -> 1 -> diagonal -> alignment of symbols
    #             bit 2 -> 2 -> left     -> gap in first sequence
    #             bit 3 -> 4 -> top      -> gap in second sequence
    trace_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.uint8)
    score_table[:,0] = -np.arange(0, len(seq1)+1)
    score_table[0,:] = -np.arange(0, len(seq2)+1)
    trace_table[1:,0] = 4
    trace_table[0,1:] = 2
    code1 = seq1.code
    code2 = seq2.code
    
    if uses_c_extensions() \
        and code1.dtype == np.uint8 \
        and code2.dtype == np.uint8:
            fill_func  = c_fill_align_table
            trace_func = c_follow_trace
    else:
            fill_func  = _fill_align_table
            trace_func = _follow_trace
    
    # Fill table
    fill_func(code1, code2, matrix.get_matrix(),
              score_table, trace_table,
              gap_opening, gap_extension,
              local = False)
    
    # Traceback
    trace_list = []
    # The start point is the last element in the table
    i = score_table.shape[0] -1
    j = score_table.shape[1] -1
    max_score = score_table[i,j]

    _traceback(trace_table, trace_list, i, j, trace_func)
    
    return [Alignment([seq1, seq2], trace, max_score) for trace in trace_list]


def align_local(seq1, seq2, matrix, gap_opening=-3, gap_extension=-1):
    """
    Perform a local alignment based on the Smith–Waterman algorithm
    algorithm [1]_.
    
    This algorithm yields an optimal alignment, i.e. the sequences
    are aligned in the way that results in the highest similarity score.
    This operation can be very time and space consuming, because both
    scale linearly with each sequence length.
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_opening : int, optional
        Negative value which is used as penalty for gap openings.
    gap_extension : int, optional
        Negative value which is used as penalty for gap extensions.
    
    Returns
    -------
    alignments : list, type=Alignment
        A list of alignments. Each alignment in the list has
        the same maximum similarity score.
    
    References
    ----------
    
    .. [1] TF Smith, MS Waterman,
       "Identification of common molecular subsequences."
       J Mol Biol, 147, 195-197 (1981).
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("CGAATACGCTTGCTCC")
    >>> seq2 = NucleotideSequence("TATAGGCGCAGCTGG")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_local(seq1, seq2, matrix,
    ...                    gap_opening=-6, gap_extension=-2)
    >>> for a in ali:
    ...     print(a, "\\n")
    ATA--CGCTTGCT
    ATAGGCGCA-GCT 
    <BLANKLINE>
    ATA--CGCTTGCT
    ATAGGCGC-AGCT
    <BLANKLINE>
    """
    # Check matrix alphabets
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("Substitution matrix is inappropriate "
                             "for the given sequences")
    # Procedure is analogous to align_global()
    score_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
    trace_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.uint8)
    code1 = seq1.code
    code2 = seq2.code
    
    if uses_c_extensions() \
        and code1.dtype == np.uint8 \
        and code2.dtype == np.uint8:
            fill_func  = c_fill_align_table
            trace_func = c_follow_trace
    else:
            fill_func  = _fill_align_table
            trace_func = _follow_trace
    
    # Fill table
    fill_func(code1, code2, matrix.get_matrix(),
              score_table, trace_table,
              gap_opening, gap_extension,
              local = True)
    
    # Traceback
    trace_list = []
    # The start point is the maximal score in the table
    i, j = np.unravel_index(np.argmax(score_table), score_table.shape)
    max_score = score_table[i,j]
    
    _traceback(trace_table, trace_list, i, j, trace_func)
    
    return [Alignment([seq1, seq2], trace, max_score) for trace in trace_list]
    
    
def _traceback(trace_table, trace_list, i_start, j_start, trace_func):
    # Pessimistic array allocation
    trace = np.full(( i_start+1 + j_start+1, 2 ), -1, dtype=np.int64)
    
    trace_func(trace_table, i_start, j_start, 0, trace, trace_list)
    
    # Replace gap entries with -1
    for i, trace in enumerate(trace_list):
        trace = np.flip(trace, axis=0)
        gap_filter = np.zeros(trace.shape, dtype=bool)
        gap_filter[np.unique(trace[:,0], return_index=True)[1], 0] = True
        gap_filter[np.unique(trace[:,1], return_index=True)[1], 1] = True
        trace[~gap_filter] = -1
        trace_list[i] = trace


def _fill_align_table(code1, code2, matrix, score_table, trace_table,
                      gap_opening, gap_extension, local):
    max_i = score_table.shape[0]
    max_j = score_table.shape[1]
    i = 1
    while i < max_i:
        j = 1
        while j < max_j:
            # Evaluate score from diagonal direction
            from_diag = score_table[i-1, j-1]
            # -1 is necessary due to the shift of the sequences
            # to the bottom/right in the table
            from_diag += matrix[code1[i-1], code2[j-1]]
            # Evaluate score from left direction
            from_left = score_table[i, j-1]
            if trace_table[i, j-1] & 2:
                from_left += gap_extension
            else:
                from_left += gap_opening
            # Evaluate score from top direction
            from_top = score_table[i-1, j]
            if trace_table[i-1, j] & 4:
                from_top += gap_extension
            else:
                from_top += gap_opening
            
            # Find maximum
            if from_diag > from_left:
                if from_diag > from_top:
                    trace, score = 1, from_diag
                elif from_diag == from_top:
                    trace, score = 5, from_diag
                else:
                    trace, score = 4, from_top
            elif from_diag == from_left:
                if from_diag > from_top:
                    trace, score = 3, from_diag
                elif from_diag == from_top:
                    trace, score = 7, from_diag
                else:
                    trace, score =  4, from_top
            else:
                if from_left > from_top:
                    trace, score = 2, from_left
                elif from_left == from_top:
                    trace, score = 6, from_diag
                else:
                    trace, score = 4, from_top
                    
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True and score <= 0:
                score_table[i,j] = 0
            else:
                score_table[i,j] = score
                trace_table[i,j] = trace
            j +=1
        i += 1


def _follow_trace(trace_table, i, j, pos, trace, trace_list):
    while trace_table[i,j] != 0:
        # -1 is necessary due to the shift of the sequences
        # to the bottom/right in the table
        trace[pos, 0] = i-1
        trace[pos, 1] = j-1
        pos += 1
        # Traces may split
        next_indices = []
        trace_value = trace_table[i,j]
        if trace_value & 1:
            next_indices.append((i-1, j-1))
        if trace_value & 2:
            next_indices.append((i, j-1))
        if trace_value & 4:
            next_indices.append((i-1, j))
        # Trace split -> Recursive call of _follow_trace() for indices[1:]
        for k in range(1, len(next_indices)):
            new_i, new_j = next_indices[k]
            _follow_trace(trace_table, new_i, new_j, pos,
                          np.copy(trace), trace_list)
        # Continue in this method with indices[0]
        i, j = next_indices[0]
    # Trim trace to correct size (delete all -1 entries)
    # and append to trace_list
    trace_list.append((trace[trace[:,0] != -1]))
