# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_local_ungapped"]

cimport cython
cimport numpy as np
from .tracetable cimport follow_trace, get_trace_linear, get_trace_affine

import itertools
import numpy as np
from .matrix import SubstitutionMatrix
from ..sequence import Sequence
from .alignment import Alignment


ctypedef np.int32_t int32
ctypedef np.int64_t int64
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64

ctypedef fused CodeType1:
    uint8
    uint16
    uint32
    uint64
ctypedef fused CodeType2:
    uint8
    uint16
    uint32
    uint64


# See tracetable.pyx for more information
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
DEF NO_STATE       = 0
DEF MATCH_STATE    = 1
DEF GAP_LEFT_STATE = 2
DEF GAP_TOP_STATE  = 3


DEF INIT_SIZE = 100


def align_local_ungapped(seq1, seq2, matrix, seed, int32 threshold,
                         str direction="both", bint score_only=False, bint check_matrix=True):
    """
    align_local_ungapped(seq1, seq2, matrix, seed, threshold,
                         direction="both", score_only=False)
    
    Perform a local alignment starting at the given `seed` position
    without introducing gaps.

    The alignment extends into one or both directions (controlled by
    `direction`) until the total alignment score falls more than
    `threshold` below the maximum score found.
    The returned alignment contains the range that yielded the maximum
    score.
    
    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
        The sequences do not need to have the same alphabets, as long as
        the two alphabets of `matrix` extend the alphabets of the two
        sequences.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    seed : tuple(int, int)
        The indices in `seq1` and `seq2` where the local alignment
        starts.
        The indices must be non-negative.
    threshold : int
        If the current score falls this value below the maximum score
        found, the alignment terminates.
    direction : {'both', 'upstream', 'downstream'}, optional
        Controls in which direction the alignment extends starting
        from the seed.
        If ``'upstream'``, the alignment starts before the `seed` and
        ends at the `seed`.
        If ``'downstream'``, the alignment starts at the `seed` and
        ends behind the `seed`.
        If ``'both'`` (default) the alignment starts before the `seed`
        and ends behind the `seed`.
        The `seed` position itself is always included in the alignment.
    score_only : bool, optional
        If set to ``True``, only the similarity score is returned
        instead of the :class:`Alignment`, decreasing the runtime
        substantially.
    
    Returns
    -------
    alignment : Alignment
        The resulting ungapped alignment.
        Only returned, if `score_only` is ``False``.
    score : int
        The alignment similarity score.
        Only returned, if `score_only` is ``True``.
    
    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("PYRRHQTITE")
    >>> matrix = SubstitutionMatrix.std_protein_matrix()
    >>> alignment = align_local_ungapped(seq1, seq2, matrix, seed=(4,7), threshold=10)
    >>> print(alignment)
    QTITE
    QTITE
    >>> alignment = align_local_ungapped(seq1, seq2, matrix, (4,7), 10, direction="upstream")
    >>> print(alignment)
    QTI
    QTI
    >>> alignment = align_local_ungapped(seq1, seq2, matrix, (4,7), 10, direction="downstream")
    >>> print(alignment)
    ITE
    ITE
    >>> score = align_local_ungapped(seq1, seq2, matrix, (4,7), 10, score_only=True)
    >>> print(score)
    24
    """
    if check_matrix:
        if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
            or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
                raise ValueError(
                    "The sequences' alphabets do not fit the matrix"
                )
    cdef const int32[:,:] score_matrix = matrix.score_matrix()
    
    cdef bint upstream
    cdef bint downstream
    if direction == "both":
        upstream = True
        downstream = True
    elif direction == "upstream":
        upstream = True
        downstream = False
    elif direction == "downstream":
        upstream = False
        downstream = True
    else:
        raise ValueError(f"Direction '{direction}' is invalid")
    
    if threshold < 0:
        raise ValueError("The threshold value must be a non-negative integer")
    
    cdef int seq1_start, seq2_start
    seq1_start, seq2_start = seed

    cdef np.ndarray code1 = seq1.code
    cdef np.ndarray code2 = seq2.code
    # For faster dispatch of the '_seed_extend()' function
    # specialization for the common case
    # This gives significant performance increase since the
    # seed extend itself runs fast
    cdef bint both_uint8 = (code1.dtype == np.uint8) \
                         & (code2.dtype == np.uint8)

    cdef int32 length
    cdef int start_offset = 0
    cdef int stop_offset = 1
    cdef int32 score
    cdef int32 total_score = 0

    # Separate alignment into two parts:
    # the regions upstream and downstream from the seed position
    # Range check to avoid negative indices
    if upstream and seq1_start > 0 and seq2_start > 0:
        # For the upstream region the respective part of the sequence
        # must be reversed
        if both_uint8:
            score, length = _seed_extend[np.uint8, np.uint8](
                code1[seq1_start-1::-1], code2[seq2_start-1::-1],
                score_matrix, threshold
            )
        else:
            score, length = _seed_extend(
                code1[seq1_start-1::-1], code2[seq2_start-1::-1],
                score_matrix, threshold
            )
        total_score += score
        start_offset -= length
    if downstream:
        if both_uint8:
            score, length = _seed_extend[np.uint8, np.uint8](
                code1[seq1_start+1:], code2[seq2_start+1:],
                score_matrix, threshold
            )
        else:
            score, length = _seed_extend(
                code1[seq1_start+1:], code2[seq2_start+1:],
                score_matrix, threshold
            )
        total_score += score
        stop_offset += length
    total_score += score_matrix[code1[seq1_start], code2[seq2_start]]
    
    if score_only:
        return total_score
    else:
        trace = np.stack([
            np.arange(seq1_start + start_offset, seq1_start + stop_offset),
            np.arange(seq2_start + start_offset, seq2_start + stop_offset)
        ], axis=-1)
    return Alignment([seq1, seq2], trace, total_score)


@cython.boundscheck(False)
@cython.wraparound(False)
def _seed_extend(CodeType1[:] code1 not None, CodeType2[:] code2 not None,
                 const int32[:,:] matrix not None, int32 threshold):
    """
    Align two sequences without introduction of gaps beginning from
    start of the given sequences.
    If the score drops too low, terminate the alignment.
    Return the similarity score and the number of aligned symbols.
    """
    cdef int i
    cdef int32 score = 0, max_score = 0
    cdef int i_max_score = -1

    # Iterate over the symbols in both sequences
    # The alignment automatically terminates,
    # if the the end of either sequence is reached
    for i in range(_min(code1.shape[0], code2.shape[0])):
        score += matrix[code1[i], code2[i]]
        if score >= max_score:
            max_score = score
            i_max_score = i
        elif max_score - score > threshold:
            # Score drops too low -> terminate alignment
            break

    # Return the total score and the number of aligned symbols at the
    # point with maximum total score
    return max_score, i_max_score + 1




def align_local_gapped(seq1, seq2, matrix, seed, int32 threshold,
                       gap_penalty=-10, max_number=1,
                       direction="both", score_only=False):
    """
    align_local_gapped(seq1, seq2, matrix, seed, threshold,
                       gap_penalty=-10, max_number=1,
                       direction="both", score_only=False)
    """
    # Check matrix alphabets
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    score_matrix = matrix.score_matrix()
    
    # Check if gap penalty is linear or affine
    if type(gap_penalty) == int:
        if gap_penalty >= 0:
            raise ValueError("Gap penalty must be negative")
    elif type(gap_penalty) == tuple:
        if gap_penalty[0] >= 0 or gap_penalty[1] >= 0:
                raise ValueError("Gap penalty must be negative")
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    
    # Check if max_number is reasonable
    if max_number < 1:
        raise ValueError(
            "Maximum number of returned alignments must be at least 1"
        )
    
    cdef bint upstream
    cdef bint downstream
    if direction == "both":
        upstream = True
        downstream = True
    elif direction == "upstream":
        upstream = True
        downstream = False
    elif direction == "downstream":
        upstream = False
        downstream = True
    else:
        raise ValueError(f"Direction '{direction}' is invalid")
    
    if threshold < 0:
        raise ValueError("The threshold value must be a non-negative integer")

    
    code1 = seq1.code
    code2 = seq2.code

    cdef int seq1_start, seq2_start
    seq1_start, seq2_start = seed


    cdef int32 score
    cdef int32 total_score = 0
    # Separate alignment into two parts:
    # the regions upstream and downstream from the seed position
    # Range check to avoid negative indices
    if upstream and seq1_start > 0 and seq2_start > 0:
        # For the upstream region the respective part of the sequence
        # must be reversed
        score, traces = _align_region(
            code1[seq1_start-1::-1], code2[seq2_start-1::-1],
            score_matrix, threshold, gap_penalty, max_number, score_only
        )
        total_score += score
        # Undo the sequence reversing
        upstream_traces = [trace[::-1] for trace in traces]
    if downstream:
        score, traces = _align_region(
            code1[seq1_start+1], code2[seq2_start+1],
            score_matrix, threshold, gap_penalty, max_number, score_only
        )
        total_score += score
        downstream_traces = traces
    total_score += score_matrix[code1[seq1_start], code2[seq2_start]]
    

    if score_only:
        return total_score
    else:
        if upstream and downstream:
            # Create cartesian product of upstream and downstream traces
            # Only consider max_number alignments
            traces = [
                np.concatenate([upstream_trace, [seed], downstream_trace])
                for i, (upstream_trace, downstream_trace) in zip(
                    range(max_number),
                    itertools.product(upstream_traces, downstream_traces)
                )
            ]
        elif upstream:
            traces = [
                np.concatenate([trace, [seed]]) for trace in upstream_traces
            ]
        else: # downstream
            traces = [
                np.concatenate([[seed], trace]) for trace in downstream_traces
            ]
        
        return [Alignment([seq1, seq2], trace, total_score)
                for trace in traces]


def _align_region(code1, code2, matrix, threshold, gap_penalty,
                  max_number, score_only):
    if type(gap_penalty) == int:
        affine_penalty = False
    else:
        affine_penalty = True
    
    
    
    init_size = (
        _min(len(code1)+1, INIT_SIZE),
        _min(len(code2)+1, INIT_SIZE)
    )
    trace_table = np.zeros(init_size, dtype=np.uint8)
    

    # Table filling
    ###############
    score_table = np.zeros(init_size, dtype=np.int32)
    # Set the initial (upper left) score value to 'threshold + 1',
    # to be able to use '0' as minus infinity value
    init_score = threshold + 1
    # This implementation does not initialize the entire first
    # row/column to avoid issues with premature pruning in the table
    # filling process
    score_table[0,0] = init_score
    
    if affine_penalty:
        raise NotImplementedError()
    else:
        trace_table, score_table = _fill_align_table(
            code1, code2, matrix, trace_table, score_table, gap_penalty,
            score_only
        )
    
    
    # Traceback
    ###########
    # Stores all possible traces (= possible alignments)
    # A trace stores the indices of the aligned symbols
    # in both sequences
    trace_list = []
    # Lists of trace starting indices
    i_list = np.zeros(0, dtype=int)
    j_list = np.zeros(0, dtype=int)
    # List of start states
    # State specifies the table the trace starts in
    state_list = np.zeros(0, dtype=int)
    # The start point is the maximal score in the table
    # Multiple starting points possible,
    # when duplicates of maximal score exist 
    if affine_penalty:
        raise NotImplementedError()
    else:
        max_score = np.max(score_table)
        i_list, j_list = np.where((score_table == max_score))
        # State is always 0 for linear gap penalty
        # since there is only one table
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
        trace = np.full(( i_start+1 + j_start+1, 2 ), -1, dtype=np.int64)
        curr_trace_count = 1
        follow_trace(
            trace_table, False, i_start, j_start, 0, trace, trace_list,
            state=state_start, curr_trace_count=&curr_trace_count,
            max_trace_count=max_number,
            # Diagonals are only needed for banded alignments
            lower_diag=0, upper_diag=0
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
    # 'follow_trace()', however, if multiple local alignment starts
    # are used, the number of created traces are the number of
    # starts times `max_number`
    trace_list = trace_list[:max_number]
    if score_only:
        # The initial score needs to be subtracted again,
        # since it was artificially added for convenience resaons
        return max_score - init_score
    else:
        return max_score - init_score, trace_list


def _extend_table(table, int dimension):
    if dimension == 0:
        new_shape = (table.shape[0] * 2, table.shape[1])
    else:
        new_shape = (table.shape[0], table.shape[1] * 2)
    new_table = np.zeros(new_shape, dtype=table.dtype)
    # Fill in exiisting data
    new_table[:table.shape[0], :table.shape[1]] = table
    return new_table


#@cython.boundscheck(False)
#@cython.wraparound(False)
def _fill_align_table(CodeType1[:] code1 not None,
                      CodeType2[:] code2 not None,
                      const int32[:,:] matrix not None,
                      uint8[:,:] trace_table not None,
                      int32[:,:] score_table not None,
                      int gap_penalty,
                      bint score_only):
    """
    """
    
    cdef int i, j, k
    # The range for i in the next, current and previous antidiagonal
    # that may contain valid cells
    cdef int i_min_curr=0, i_max_curr=1
    cdef int i_min_next=1, i_max_next=0
    cdef int i_min_prev=1, i_max_prev=0

    cdef int j_max_curr
    cdef int j_max_next

    cdef int32 from_diag, from_left, from_top
    cdef uint8 trace
    cdef int32 score

    # Instead of iteration over row and column,
    # iterate over antidiagonal and diagonal to achieve symmetric
    # treatment of both sequences
    for k in range(1, _max(code1.shape[0], code2.shape[0])):
        for i in range(i_min_curr, i_max_curr+1):
            j = k - i
            # Evaluate score from diagonal direction
            if i != 0 and j != 0:
                from_diag = score_table[i-1, j-1]
                if from_diag != 0:
                    # -1 in sequence index is necessary
                    # due to the shift of the sequences
                    # to the bottom/right in the table
                    from_diag += matrix[code1[i-1], code2[j-1]]
                else:
                    from_diag = 0
            else:
                from_diag = 0
            # Evaluate score through gap insertion
            if i != 0:
                from_top = score_table[i-1, j] + gap_penalty
            else:
                from_top = 0
            if j != 0:
                from_left = score_table[i, j-1] + gap_penalty
            else:
                from_left = 0
            
            trace = get_trace_linear(from_diag, from_left, from_top, &score)
            
            if score > 0:
                if i_min_curr == k:
                    i_min_curr = i
                i_max_curr = i
                score_table[i,j] = score
                trace_table[i,j] = trace
        

        # Prune index range for next antidiagonal
        # to range where valid cells exist
        i_min_next = _min(i_min_curr,     i_min_prev + 1)
        i_max_next = _max(i_max_curr + 1, i_max_prev + 1)
        # The algorithm has finished,
        # if the next antidiagonal has no range of valid cells
        if i_min_next > i_max_next:
            break

        # Expand ndarrays
        # if their size would be exceeded in the next iteration
        if i_max_next >= score_table.shape[0]:
            score_table = _extend_table(score_table, 0)
            trace_table = _extend_table(trace_table, 0)
        j_max_next = k + 1 - i_min_next
        if j_max_next >= score_table.shape[1]:
            score_table = _extend_table(score_table, 1)
            trace_table = _extend_table(score_table, 1)

        # Prepare values for next iteration
        i_min_prev = i_min_curr
        i_max_prev = i_max_curr
        i_min_curr = i_min_next
        i_max_curr = i_max_next
        # Reset values for next iteration to most 'restrictive' values
        # These restrictive values are overwritten in the next itaterion
        # if valid cells are present
        i_min_next = k+1
        i_max_next = 0
    
    j_max_curr = k - i_min_curr
    filled_range = (i_max_curr + 1, j_max_curr + 1)
    return np.asarray(trace_table)[filled_range], \
           np.asarray(score_table)[filled_range]


cdef inline int _min(int a, int b):
    return a if a < b else b

cdef inline int _max(int a, int b):
    return a if a > b else b