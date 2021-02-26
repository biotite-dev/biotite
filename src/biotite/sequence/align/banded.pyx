# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_banded"]

cimport cython
cimport numpy as np

from .matrix import SubstitutionMatrix
from ..sequence import Sequence
from .alignment import Alignment
import numpy as np


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

ctypedef fused CodeType1:
    uint8
    #uint16
    #uint32
    #uint64
ctypedef fused CodeType2:
    uint8
    #uint16
    #uint32
    #uint64


# The trace table will save the directions a cell came from
# A "1" in the corresponding bit in the trace table means
# the cell came from this direction
# Values for linear gap penalty (one score table)
#     bit 1 -> 1  -> diagonal -> alignment of symbols
#     bit 2 -> 2  -> left     -> gap in first sequence
#     bit 3 -> 4  -> top      -> gap in second sequence
# Values for affine gap penalty (three score tables)
#     bit 1 -> 1  -> match - match transition
#     bit 2 -> 2  -> seq 1 gap - match transition
#     bit 3 -> 4  -> seq 2 gap - match transition
#     bit 4 -> 8  -> match - seq 1 gap transition
#     bit 5 -> 16 -> seq 1 gap - seq 1 gap transition
#     bit 6 -> 32 -> match - seq 2 gap transition
#     bit 7 -> 64 -> seq 2 gap - seq 2 gap transition
DEF MATCH    = 1
DEF GAP_LEFT = 2
DEF GAP_TOP  = 4
DEF MATCH_TO_MATCH       = 1
DEF GAP_LEFT_TO_MATCH    = 2
DEF GAP_TOP_TO_MATCH     = 4
DEF MATCH_TO_GAP_LEFT    = 8
DEF GAP_LEFT_TO_GAP_LEFT = 16
DEF MATCH_TO_GAP_TOP     = 32
DEF GAP_TOP_TO_GAP_TOP   = 64


def align_banded(seq1, seq2, matrix, band, gap_penalty=-10, local=False,
                 max_number=1000):
    """
    align_banded(seq1, seq2, matrix, band, gap_penalty=-10, local=False,
                 max_number=1000)

    Perform a local or semi-global alignment within a defined band.
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    band : tuple(int, int)
        The diagonals that represent the lower and upper limit of the
        search space.
        A diagonal :math:`d` is defined as :math:`d = j-i`, where
        :math:`i` and :math:`j` are positions in `seq1` and `seq2`,
        respectively.
        An alignment of sequence positions where :math:`d` is lower than
        the lower limit or higher than the upper limit are ignored.
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        linear gap penalty.
        If a tuple is provided, an affine gap penalty is used.
        The first integer in the tuple is the gap opening penalty,
        the second integer is the gap extension penalty.
        The values need to be negative. (Default: *-10*)
    local : bool, optional
        If false, a global alignment is performed, otherwise a local
        alignment is performed. (Default: False)
    max_number : int, optional
        The maximum number of alignments returned.
        When the number of branches exceeds this value in the traceback
        step, no further branches are created.
        (Default: 1000)
    
    Returns
    -------
    alignments : list, type=Alignment
        A list of alignments.
        Each alignment in the list has the same maximum similarity
        score.
    
    See also
    --------
    align_optimal

    Notes
    -----
    The band with is equal to the maximum difference between the
    number of gaps in the sequences.
    This means for any poistion in the alignment, the alogrithm
    will not consider inserting a gap into a seqeunce, if this
    sequence has already `band_width` more gaps than the other
    sequence, even if inserting another gap would yield a more optimal
    alignment.
    The limited band width is the central difference between the banded
    alignment heuristic [1]_ and the optimal alignment algorithms
    [2]_[3]_.
    
    References
    ----------
    
    .. [1] WR Pearson, DJ Lipman,
       "Improved tools for biological sequence comparison."
       Proc Natl Acad Sci USA, 85, 2444–2448 (1988).
    .. [2] SB Needleman, CD Wunsch,
       "A general method applicable to the search for similarities
       in the amino acid sequence of two proteins."
       J Mol Biol, 48, 443-453 (1970).
    .. [3] TF Smith, MS Waterman,
       "Identification of common molecular subsequences."
       J Mol Biol, 147, 195-197 (1981).
    .. [4] O Gotoh,
       "An improved algorithm for matching biological sequences."
       J Mol Biol, 162, 705-708 (1982).
    """
    # Check matrix alphabets
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    # Check if gap penalty is linear or affine
    if type(gap_penalty) == int:
        if gap_penalty > 0:
            raise ValueError("Gap penalty must be negative")
        affine_penalty = False
    elif type(gap_penalty) == tuple:
        if gap_penalty[0] > 0 or gap_penalty[1] > 0:
                raise ValueError("Gap penalty must be negative")
        affine_penalty = True
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    # Check if max_number is reasonable
    if max_number < 1:
        raise ValueError(
            "Maximum number of returned alignments must be at least 1"
        )
    
    # The shorter sequence is the one on the left of the matrix
    # -> shorter sequence is 'seq1'
    if len(seq2) < len(seq1):
        seq1, seq2 = seq2, seq1
        band = [-diag for diag in band]
        matrix = matrix.transpose()
        is_swapped = True
    else:
        is_swapped = False
    lower_diag, upper_diag = min(band), max(band)
    band_width = upper_diag - lower_diag + 1
    if band_width < 1:
        raise ValueError("The width of the band is 0")
    if len(seq1) + upper_diag <= 0 or lower_diag >= len(seq2):
        raise ValueError(
            "Alignment band is out of range, the band allows no overlap "
            "between both sequences"
        )
    # Crop band diagonals to reasonable size, so that it at maximum
    # covers the search space of an unbanded alignment
    lower_diag = max(lower_diag, -len(seq1)+1)
    upper_diag = min(upper_diag,  len(seq2)-1)

    # This implementation uses transposed tables in comparison
    # to the common visualization
    # This means, the first sequence is one the left
    # and the second sequence is at the top

    # Terminal gap column on the left can be omitted in this algorithm,
    # as terminal gaps are not part of the alignment
    # This is not possible for the top row, as the dynamic programming
    # algorithm requires these initial values
    # On the left and right side an additional column is inserted
    # representing the invalid boundaries of the band
    # This prevents unnecessary bound checks when filling the dynamic
    # programming matrix (score and trace)
    trace_table = np.zeros((len(seq1)+1, band_width+2), dtype=np.uint8)
    code1 = seq1.code
    code2 = seq2.code
    

    # Table filling
    ###############
    
    # A score value that signals that the respective direction in the
    # dynamic programming matrix should not be used since, it would be
    # outside the band
    # It is the 'worst' score available, so the trace table will never
    # include such a direction
    neg_inf = np.iinfo(np.int32).min
    # Correct the 'negative infinity' integer, by making it more positve
    # This prevents an integer underflow when the gap penalty or
    # match score is added to this value
    neg_inf -= min(gap_penalty) if affine_penalty else gap_penalty
    min_score = np.min(matrix.score_matrix())
    if min_score < 0:
        neg_inf -= min_score
    ###
    neg_inf = -1000
    np.set_printoptions(edgeitems=1000, linewidth=100000)
    ###

    if affine_penalty:
        pass
        """
        # Affine gap penalty
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
        # Value for negative infinity
        # Used to prevent unallowed state transitions
        # in the first row and column
        # Subtraction of gap_open, gap_ext and lowest score value
        # to prevent integer overflow
        neg_inf = np.iinfo(np.int32).min - 2*gap_open - 2*gap_ext
        min_score = np.min(matrix.score_matrix())
        if min_score < 0:
            neg_inf -= min_score
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
            if terminal_penalty:
                # Terminal gaps are penalized
                # -> Penalties in first row/column
                g1_table[0, 1:] = (np.arange(len(seq2)) * gap_ext) + gap_open
                g2_table[1:,0 ] = (np.arange(len(seq1)) * gap_ext) + gap_open
            trace_table[1:,0] = 64
            trace_table[0,1:] = 16
        _fill_align_table_affine(code1, code2,
                                 matrix.score_matrix(), trace_table,
                                 m_table, g1_table, g2_table,
                                 gap_open, gap_ext, terminal_penalty, local)
        """
    else:
        # Linear gap penalty

        # As explained for the trace table (see above),
        # the score table is filled with with INVALID values on the left
        # and right column
        score_table = np.zeros((len(seq1)+1, band_width+2), dtype=np.int32)
        score_table[:,  0] = neg_inf
        score_table[:, -1] = neg_inf
        _fill_align_table(
            code1, code2, matrix.score_matrix(), trace_table, score_table,
            lower_diag, upper_diag, gap_penalty, local
        )
    

    # Traceback
    ###########

    #print(score_table)
    #print()
    #print(trace_table)
    #print()

    # Stores all possible traces (= possible alignments)
    # A trace stores the indices of the aligned symbols
    # in both sequences
    trace_list = []
    # Lists of trace starting indices
    i_list = np.zeros(0, dtype=int)
    j_list = np.zeros(0, dtype=int)
    # List of start states
    # State specifies the table the trace starts in
    # 0 -> linear gap penalty, only one table
    # 1 -> m
    # 2 -> g1
    # 3 -> g2
    state_list = np.zeros(0, dtype=int)
    if local:
        # The start point is the maximal score in the table
        # Multiple starting points possible,
        # when duplicates of maximum score exist
        if affine_penalty:
            pass
            """
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
            """
        else:
            max_score = np.max(score_table)
            i_list, j_list = np.where((score_table == max_score))
            # State is always 0 for linear gap penalty
            # since there is only one table
            state_list = np.zeros(len(i_list), dtype=int)
    else:
        # Get all allowed trace start indices
        possible_i_start, possible_j_start = get_global_trace_starts(
            len(seq1), len(seq2), lower_diag, upper_diag
        )
        if affine_penalty:
            pass
            """
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
            """
        else:
            # Choose the trace start index with the highest score
            # in the score table
            scores = score_table[possible_i_start, possible_j_start]
            max_score = np.max(scores)
            best_indices = np.where(scores == max_score)
            i_list = possible_i_start[best_indices]
            j_list = possible_j_start[best_indices]
            state_list = np.zeros(len(i_list), dtype=int)
    
    # Follow the traces specified in state and indices lists
    cdef int curr_trace_count
    for k in range(len(i_list)):
        i_start = i_list[k]
        j_start = j_list[k]
        state_start = state_list[k]
        # Pessimistic array allocation:
        # The maximum trace length arises from an alignment, where each
        # symbol is aligned to a gap
        trace = np.full((len(seq1) + len(seq2), 2), -1, dtype=np.int64)
        curr_trace_count = 1
        _follow_trace(
            trace_table, lower_diag, upper_diag, i_start, j_start, 0,
            trace, trace_list, state=state_start,
            curr_trace_count=&curr_trace_count, max_trace_count=max_number
        )
    
    # Replace gap entries in trace with -1
    for i, trace in enumerate(trace_list):
        trace = np.flip(trace, axis=0)
        gap_filter = np.zeros(trace.shape, dtype=bool)
        gap_filter[np.unique(trace[:,0], return_index=True)[1], 0] = True
        gap_filter[np.unique(trace[:,1], return_index=True)[1], 1] = True
        trace[~gap_filter] = -1
        trace_list[i] = trace

    # Limit the number of generated alignments to `max_number`:
    # In most cases this is achieved by discarding branches in
    # '_follow_trace()', however, if multiple alignment starts
    # are used, the number of created traces are the number of
    # starts times `max_number`
    trace_list = trace_list[:max_number]
    if is_swapped:
        return [Alignment([seq2, seq1], np.flip(trace, axis=1), max_score)
                for trace in trace_list]
    else:
        return [Alignment([seq1, seq2], trace, max_score)
                for trace in trace_list]


@cython.boundscheck(False)
@cython.wraparound(False)
def _fill_align_table(CodeType1[:] code1 not None,
                      CodeType2[:] code2 not None,
                      const int32[:,:] mat not None,
                      uint8[:,:] trace_table not None,
                      int32[:,:] score_table not None,
                      int lower_diag,
                      int upper_diag,
                      int gap_penalty,
                      bint local):
    """
    Fill an alignment table with linear gap penalty using dynamic
    programming.

    Parameters
    ----------
    code1, code2
        The sequence code of each sequence to be aligned.
    mat
        The score matrix obtained from the :class:`SubstitutionMatrix`
        object.
    trace_table
        A matrix containing values indicating the direction for the
        traceback step.
        The matrix is filled in this function
    score_table
        The alignment table.
        The matrix is filled in this function.
    gap_penalty
        The linear gap penalty.
    term_penalty
        Indicates, whether terminal gaps should be penalized.
    local
        Indicates, whether a local alignment should be performed.
    """
    
    cdef int i, j
    cdef int seq_i, seq_j
    cdef int max_i, max_j
    cdef int32 from_diag, from_left, from_top
    cdef uint8 trace
    cdef int32 score
    
    # Used in case terminal gaps are not penalized
    i_max = score_table.shape[0] -1
    j_max = score_table.shape[1] -1

    # Starts at 1 since the first row and column are already filled
    for seq_i in range(0, code1.shape[0]):
        # Transform sequence index into table index
        i = seq_i + 1
        for seq_j in range(
            max(0,              seq_i + lower_diag),
            min(code2.shape[0], seq_i + upper_diag+1)
        ):
            # Transform sequence index into table index
            # Due to the diagonal band and its 'straightening'
            # seq_j must be transformed to obtain the table index
            j = seq_j - seq_i - lower_diag + 1

            # Evaluate score from available directions:
            # Due the 'straightening' of the the diagonal band,
            # the 'upper left' and 'upper' direction from the classic
            # matrix become 'upper' and 'upper right', respectively
            from_diag = score_table[i-1, j  ] + mat[code1[seq_i], code2[seq_j]]
            from_left = score_table[i,   j-1] + gap_penalty
            from_top  = score_table[i-1, j+1] + gap_penalty
            
            # Find maximum
            if from_diag > from_left:
                if from_diag > from_top:
                    trace = MATCH
                    score = from_diag
                elif from_diag == from_top:
                    trace = MATCH | GAP_TOP
                    score = from_diag
                else:
                    trace = GAP_TOP
                    score = from_top
            elif from_diag == from_left:
                if from_diag > from_top:
                    trace = MATCH | GAP_LEFT
                    score = from_diag
                elif from_diag == from_top:
                    trace = MATCH | GAP_LEFT | GAP_TOP
                    score = from_diag
                else:
                    trace = GAP_TOP
                    score = from_top
            else:
                if from_left > from_top:
                    trace = GAP_LEFT
                    score = from_left
                elif from_left == from_top:
                    trace = GAP_LEFT | GAP_TOP
                    score = from_left
                else:
                    trace = GAP_TOP
                    score = from_top
            
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True and score <= 0:
                score_table[i,j] = 0
            else:
                score_table[i,j] = score
                trace_table[i,j] = trace


def get_global_trace_starts(seq1_len, seq2_len, lower_diag, upper_diag):
    band_width = upper_diag - lower_diag + 1
    
    j = np.arange(1, band_width + 1)
    seq_j = j + (seq1_len-1) + lower_diag - 1
    # Start from the end from the first (shorter) sequence,
    # if the table cell is in bounds of the second (longer) sequence,
    # otherwise start from the end of the second sequence
    i = np.where(
        seq_j < seq2_len,
        np.full(len(j), (seq1_len-1) + 1, dtype=int),
        # Take:
        #
        # seq_j = j + (seq1_len-1) + lower_diag - 1
        #
        # Replace seq_j with last sequence position of second sequence
        # and last sequence position of first sequence with seq_i:
        #
        # (seq2_len-1) = j + seq_i + lower_diag - 1
        #
        # Replace seq_i with corresponding i in trace table:
        #
        # (seq2_len-1) = j + (i - 1) + lower_diag - 1
        #
        # Resolve to i:
        #
        (seq2_len-1) - j - lower_diag + 2
    )
    return i, j



cdef int _follow_trace(uint8[:,:] trace_table,
                        int lower_diag, int upper_diag,
                        int i, int j, int pos,
                        int64[:,:] trace,
                        list trace_list,
                        int state,
                        int* curr_trace_count,
                        int max_trace_count) except -1:
    """
    Calculate traces from a trace table.

    Parameters
    ----------
    trace_table
        A matrix containing values indicating the direction for the
        traceback.
    i, j
        The current position in the trace table.
        For the first branch, this is the start of the traceback.
        For additional branches this is the start of the respective
        branch.
    pos
        The index in the alignment trace to be created.
        For the first branch, this is 0.
        For additional branches the value of the parent branch is taken.
    trace
        The alignment trace to be filled
    trace_list
        When a trace is finished, it is appened to this list
    state
        The current table (m, g1, g2) the traceback is in, taken from
        parent branch. Always 0 when a linear gap penalty is used.
    curr_trace_count
        The current number of branches. The value is a pointer, so that
        updating this value propagates the value to all other branches
    max_trace_count
        The maximum number of branches created. When the number of
        branches reaches this value, no new branches are created.
    """
    
    cdef list next_indices
    cdef list next_states
    cdef int trace_value
    cdef int k
    cdef int seq_i, seq_j
    
    if state == 0:
        # linear gap penalty
        while trace_table[i,j] != 0:
            seq_i = i - 1
            seq_j = j + seq_i + lower_diag - 1
            trace[pos, 0] = seq_i
            trace[pos, 1] = seq_j
            pos += 1
            # Traces may split
            next_indices = []
            trace_value = trace_table[i,j]
            if trace_value & MATCH:
                next_indices.append((i-1, j  ))
            if trace_value & GAP_LEFT:
                next_indices.append((i,   j-1))
            if trace_value & GAP_TOP:
                next_indices.append((i-1, j+1))
            # Trace branching
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                if curr_trace_count[0] < max_trace_count:
                    curr_trace_count[0] += 1
                    new_i, new_j = next_indices[k]
                    _follow_trace(
                        trace_table, lower_diag, upper_diag, new_i, new_j, pos,
                        np.copy(trace), trace_list, 0, curr_trace_count,
                        max_trace_count
                    )
            # Continue in this method with indices[0]
            i, j = next_indices[0]
    """
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
            # Trace branching
            # -> Recursive call of _follow_trace() for indices[1:]
            for k in range(1, len(next_indices)):
                if curr_trace_count[0] < max_trace_count:
                    curr_trace_count[0] += 1
                    new_i, new_j = next_indices[k]
                    new_state = next_states[k]
                    _follow_trace(trace_table, new_i, new_j, pos,
                                np.copy(trace), trace_list, new_state,
                                curr_trace_count, max_trace_count)
            # Continue in this method with indices[0] and states[0]
            i, j = next_indices[0]
            state = next_states[0]
    """
    # Trim trace to correct size (delete all pure -1 entries)
    # and append to trace_list
    tr_arr = np.asarray(trace)
    trace_list.append(tr_arr[(tr_arr[:,0] != -1) | (tr_arr[:,1] != -1)])
    return 0
