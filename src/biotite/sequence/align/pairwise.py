# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

from ...extensions import has_c_extensions, uses_c_extensions
from .matrix import SubstitutionMatrix
from ..sequence import Sequence
from .alignment import Alignment
import numpy as np
import copy
import textwrap
if has_c_extensions():
    from .calign import c_fill_align_table, c_fill_align_table_affine, \
                        c_follow_trace
    

__all__ = ["simple_score", "align_optimal"]


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


def align_optimal(seq1, seq2, matrix, gap_penalty=-10, local=False):
    """
    Perform an optimal alignment of two sequences based on the
    dynamic programming algorithm [1]_.
    
    This algorithm yields an optimal alignment, i.e. the sequences
    are aligned in the way that results in the highest similarity
    score. This operation can be very time and space consuming,
    because both scale linearly with each sequence length.
    
    The aligned sequences do not need to be instances from the same
    `Sequence` subclass, since they do not need to have the same
    alphabet. The only requirement is that the substitution matrix'
    alphabets extend the alphabets of the two sequences.
    
    This function can either perform a global alignment, based on the
    Needleman-Wunsch algorithm [1]_ or a local alignment, based on the
    Smith–Waterman algorithm [2]_.
    
    Furthermore this function supports affine gap penalties using the
    Gotoh algorithm [3]_, however, this requires approximately 4 times
    the RAM space and execution time.
    
    This function optionally uses C-extensions. 
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_penalty : int or (tuple, dtype=int), optional
        If an integer is provided, the value will be interpreted as
        general gap penalty. If a tuple is provided, an affine gap
        penalty is used. The first integer in the tuple is the gap
        opening penalty, the second integer is the gap extension
        penalty.
        The values need to be negative. (Default: *-10*)
    local : bool, optional
        If false, a global alignment is performed, otherwise a local
        alignment is performed. (Default: False)
    
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
    .. [2] TF Smith, MS Waterman,
       "Identification of common molecular subsequences."
       J Mol Biol, 147, 195-197 (1981).
    .. [3] O Gotoh,
       "An improved algorithm for matching biological sequences."
       J Mol Biol, 162, 705-708 (1982).
    
    Examples
    --------
    
    >>> seq1 = NucleotideSequence("ATACGCTTGCT")
    >>> seq2 = NucleotideSequence("AGGCGCAGCT")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> ali = align_optimal(seq1, seq2, matrix, gap_penalty=-6)
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
    # Check if gap penalty is gernal or affine
    if type(gap_penalty) == int:
        affine_penalty = False
    elif type(gap_penalty) == tuple:
        affine_penalty = True
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    # This implementation uses transposed tables in comparison
    # to the common implementation
    # Therefore the first sequence is one the left
    # and the second sequence is at the top
    
    # The table saving the directions a field came from
    # A "1" in the corresponding bit in the trace table means
    # the field came from this direction
    # Values for general gap penalty (one score table)
    #     bit 1 -> 1  -> diagonal -> alignment of symbols
    #     bit 2 -> 2  -> left     -> gap in first sequence
    #     bit 3 -> 4  -> top      -> gap in second sequence
    # Values for affine gap penalty (three score table)
    #     bit 1 -> 1  -> match - match transition
    #     bit 2 -> 2  -> seq 1 gap - match transition
    #     bit 3 -> 4  -> seq 2 gap - match transition
    #     bit 4 -> 8  -> match - seq 1 gap transition
    #     bit 5 -> 16 -> seq 1 gap - seq 1 gap transition
    #     bit 6 -> 32 -> match - seq 2 gap transition
    #     bit 7 -> 64 -> seq 2 gap - seq 2 gap transition
    trace_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.uint8)
    code1 = seq1.code
    code2 = seq2.code
    
    if uses_c_extensions():
            fill_func = c_fill_align_table
            fill_func_affine = c_fill_align_table_affine
            trace_func = c_follow_trace
    else:
            fill_func = _fill_align_table
            fill_func_affine = _fill_align_table_affine
            trace_func = _follow_trace
    
    # Table filling
    ###############
    if affine_penalty:
        # Affine gap penalty
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
        neg_inf = np.iinfo(np.int32).min - 2*gap_open  - 2*gap_ext
        # m_table, g1_table and g2_table are the 3 score tables
        m_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
        g1_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
        g2_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
        m_table [0 ,1:] = neg_inf
        m_table [1:,0 ] = neg_inf
        g1_table[:, 0 ] = neg_inf
        g2_table[0, : ] = neg_inf
        # Initialize first row and column for global alignments
        if not local:
            g1_table[0, 1:] = (np.arange(len(seq2)) * gap_ext) + gap_open
            g2_table[1:,0 ] = (np.arange(len(seq1)) * gap_ext) + gap_open
            trace_table[1:,0] = 64
            trace_table[0,1:] = 16
        fill_func_affine(code1, code2, matrix.score_matrix(), trace_table,
                         m_table, g1_table, g2_table,
                         gap_open, gap_ext, local)
    else:
        # General gap penalty
        # The table for saving the scores
        score_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
        # Initialize first row and column for global alignments
        if not local:
            score_table[:,0] = np.arange(len(seq1)+1) * gap_penalty
            score_table[0,:] = np.arange(len(seq2)+1) * gap_penalty
            trace_table[1:,0] = 4
            trace_table[0,1:] = 2
        fill_func(code1, code2, matrix.score_matrix(), trace_table,
                  score_table,
                  gap_penalty, local)
    
    # Traceback
    ###########
    # Stores all possible traces (= possible alignments)
    # A trace stores the indices of the aligned symbols
    # in both sequences
    trace_list = []
    # Lists of trace staring indices
    i_list = np.zeros(0, dtype=int)
    j_list = np.zeros(0, dtype=int)
    # List of start states
    # State specifies the table the trace starts in
    # 0 -> general gap penalty, only one table
    # 1 -> m
    # 2 -> g1
    # 3 -> g2
    state_list = np.zeros(0, dtype=int)
    if local:
        # The start point is the maximal score in the table
        # Multiple starting points possible,
        # when duplicates of maximal score exist 
        if affine_penalty:
            max_score = np.max([m_table, g1_table, g2_table])
            # Start indices in m_table
            i_list_new, j_list_new = np.where((m_table == max_score))
            i_list = np.append(i_list, i_list_new)
            j_list = np.append(j_list, j_list_new)
            state_list = np.append(state_list, np.full(len(i_list_new), 1))
            # Start indices in g1_table
            i_list_new, j_list_new = np.where((g1_table == max_score))
            i_list = np.append(i_list, i_list_new)
            j_list = np.append(j_list, j_list_new)
            state_list = np.append(state_list, np.full(len(i_list_new), 2))
            # Start indices in g2_table
            i_list_new, j_list_new = np.where((g2_table == max_score))
            i_list = np.append(i_list, i_list_new)
            j_list = np.append(j_list, j_list_new)
            state_list = np.append(state_list, np.full(len(i_list_new), 3))
        else:
            max_score = np.max(score_table)
            i_list, j_list = np.where((score_table == max_score))
            # State is always 0 for general gap penalty
            # since there is only one table
            state_list = np.zeros(len(i_list), dtype=int)
    else:
        # The start point is the last element in the table
        # -1 in start indices due to sequence offset mentioned before
        i_start = trace_table.shape[0] -1
        j_start = trace_table.shape[1] -1
        if affine_penalty:
            max_score = max(m_table[i_start,j_start],
                            g1_table[i_start,j_start],
                            g2_table[i_start,j_start])
            if m_table[i_start,j_start] == max_score:
                i_list = np.append(i_list, i_start)
                j_list = np.append(j_list, j_start)
                state_list = np.append(state_list, 1)
            if g1_table[i_start,j_start] == max_score:
                i_list = np.append(i_list, i_start)
                j_list = np.append(j_list, j_start)
                state_list = np.append(state_list, 2)
            if g2_table[i_start,j_start] == max_score:
                i_list = np.append(i_list, i_start)
                j_list = np.append(j_list, j_start)
                state_list = np.append(state_list, 3)
        else:
            i_list = np.append(i_list, i_start)
            j_list = np.append(j_list, j_start)
            state_list = np.append(state_list, 0)
            max_score = score_table[i_start,j_start]
    # Follow the traces specified in state and indices lists
    for k in range(len(i_list)):
        i_start = i_list[k]
        j_start = j_list[k]
        state_start = state_list[k]
        # Pessimistic array allocation
        trace = np.full(( i_start+1 + j_start+1, 2 ), -1, dtype=np.int64)
        trace_func(trace_table, i_start, j_start, 0, trace, trace_list,
                   state=state_start)
    
    # Replace gap entries in trace with -1
    for i, trace in enumerate(trace_list):
        trace = np.flip(trace, axis=0)
        gap_filter = np.zeros(trace.shape, dtype=bool)
        gap_filter[np.unique(trace[:,0], return_index=True)[1], 0] = True
        gap_filter[np.unique(trace[:,1], return_index=True)[1], 1] = True
        trace[~gap_filter] = -1
        trace_list[i] = trace
    
    return [Alignment([seq1, seq2], trace, max_score) for trace in trace_list]


def _fill_align_table(code1, code2, matrix, trace_table, score_table,
                      gap_penalty, local):
    for i in range(1, trace_table.shape[0]):
        for j in range(1, trace_table.shape[1]):
            # Evaluate score from diagonal direction
            from_diag = score_table[i-1, j-1]
            # -1 is necessary due to the shift of the sequences
            # to the bottom/right in the table
            from_diag += matrix[code1[i-1], code2[j-1]]
            # Evaluate score from left direction
            from_left = score_table[i, j-1] + gap_penalty
            # Evaluate score from top direction
            from_top = score_table[i-1, j] + gap_penalty
            
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


def _fill_align_table_affine(code1, code2, matrix, trace_table,
                             m_table, g1_table, g2_table,
                             gap_open, gap_ext, local):
    for i in range(1, trace_table.shape[0]):
        for j in range(1, trace_table.shape[1]):
            # Calculate the scores for possible transitions
            # into the current cell
            similarity = matrix[code1[i-1], code2[j-1]]
            mm_score = m_table[i-1,j-1] + similarity
            g1m_score = g1_table[i-1,j-1] + similarity
            g2m_score = g2_table[i-1,j-1] + similarity
            # No transition from g1_table to g2_table and vice versa
            # Since this would mean adjacent gaps in both sequences
            # A substitution makes more sense in this case
            mg1_score = m_table[i,j-1]  + gap_open
            g1g1_score = g1_table[i,j-1] + gap_ext
            mg2_score = m_table[i-1,j]  + gap_open
            g2g2_score = g2_table[i-1,j] + gap_ext
            
            # Find maximum score and trace
            # (similar to general gap method)
            # At first for match table (m_table)
            if mm_score > g1m_score:
                if mm_score > g2m_score:
                    trace, m_score = 1, mm_score
                elif mm_score == g2m_score:
                    trace, m_score = 5, mm_score
                else:
                    trace, m_score = 4, g2m_score
            elif mm_score == g1m_score:
                if mm_score > g2m_score:
                    trace, m_score = 3, mm_score
                elif mm_score == g2m_score:
                    trace, m_score = 7, mm_score
                else:
                    trace, m_score =  4, g2m_score
            else:
                if g1m_score > g2m_score:
                    trace, m_score = 2, g1m_score
                elif g1m_score == g2m_score:
                    trace, m_score = 6, mm_score
                else:
                    trace, m_score = 4, g2m_score
            #Secondly for gap tables (g1_table and g2_table)
            if mg1_score > g1g1_score:
                trace |= 8
                g1_score = mg1_score
            elif mg1_score < g1g1_score:
                trace |= 16
                g1_score = g1g1_score
            else:
                trace |= 24
                g1_score = mg1_score
            if mg2_score > g2g2_score:
                trace |= 32
                g2_score = mg2_score
            elif mg2_score < g2g2_score:
                trace |= 64
                g2_score = g2g2_score
            else:
                trace |= 96
                g2_score = g2g2_score
            # Fill values into tables
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True:
                if m_score <= 0:
                    m_table[i,j] = 0
                    # End trace in specific table
                    # by filtering the the bits of other tables  
                    trace &= ~7
                else:
                    m_table[i,j] = m_score
                if g1_score <= 0:
                    g1_table[i,j] = 0
                    trace &= ~24
                else:
                    g1_table[i,j] = g1_score
                if g2_score <= 0:
                    g2_table[i,j] = 0
                    trace &= ~96
                else:
                    g2_table[i,j] = g2_score
            else:
                m_table[i,j] = m_score
                g1_table[i,j] = g1_score
                g2_table[i,j] = g2_score
            trace_table[i,j] = trace

def _follow_trace(trace_table, i, j, pos, trace, trace_list, state):
    if state == 0:
        # General gap penalty
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
            # Trace split
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                new_i, new_j = next_indices[k]
                _follow_trace(trace_table, new_i, new_j, pos,
                              np.copy(trace), trace_list, 0)
            # Continue in this method with indices[0]
            i, j = next_indices[0]
    else:
        # Affine gap penalty -> state specifies the table
        # we are currently in
        # The states are
        # 1 -> m
        # 2 -> g1
        # 3 -> g2
        while True:
            # Check for loop break condition
            # -> trace for current table (state) is 0
            if (   (state == 1 and trace_table[i,j] & 7 == 0)
                or (state == 2 and trace_table[i,j] & 24 == 0)
                or (state == 3 and trace_table[i,j] & 96 == 0)):
                    break
            # If no break occurred, continue as usual
            trace[pos, 0] = i-1
            trace[pos, 1] = j-1
            pos += 1
            next_indices = []
            next_states = []
            # Get value of trace respective of current state
            # = table trace is currently in
            if state == 1:
                trace_value = trace_table[i,j] & 7
            elif state == 2:
                trace_value = trace_table[i,j] & 24
            else: # state == 3:
                trace_value = trace_table[i,j] & 96
            # Determine indices and state of next trace step
            if trace_value & 1:
                next_indices.append((i-1, j-1))
                next_states.append(1)
            if trace_value & 2:
                next_indices.append((i-1, j-1))
                next_states.append(2)
            if trace_value & 4:
                next_indices.append((i-1, j-1))
                next_states.append(3)
            if trace_value & 8:
                next_indices.append((i, j-1))
                next_states.append(1)
            if trace_value & 16:
                next_indices.append((i, j-1))
                next_states.append(2)
            if trace_value & 32:
                next_indices.append((i-1, j))
                next_states.append(1)
            if trace_value & 64:
                next_indices.append((i-1, j))
                next_states.append(3)
            # Trace split
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                new_i, new_j = next_indices[k]
                new_state = next_states[k]
                _follow_trace(trace_table, new_i, new_j, pos,
                              np.copy(trace), trace_list, new_state)
            # Continue in this method with indices[0] and states[0]
            i, j = next_indices[0]
            state = next_states[0]
    # Trim trace to correct size (delete all pure -1 entries)
    # and append to trace_list
    trace_list.append((trace[(trace[:,0] != -1) | (trace[:,1] != -1)]))
