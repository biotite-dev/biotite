# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_ungapped", "align_optimal"]

cimport cython
cimport numpy as np
from .tracetable cimport follow_trace, get_trace_linear, get_trace_affine, \
                         TraceDirectionLinear, TraceDirectionAffine

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
    uint16
    uint32
    uint64
ctypedef fused CodeType2:
    uint8
    uint16
    uint32
    uint64


def align_ungapped(seq1, seq2, matrix, score_only=False):
    """
    align_ungapped(seq1, seq2, matrix, score_only=False)

    Align two sequences without insertion of gaps.

    Both sequences need to have the same length.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences, whose similarity should be scored.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    score_only : bool, optional
        If true return only the score instead of an alignment.

    Returns
    -------
    score : Alignment or int
        The resulting trivial alignment. If `score_only` is set to true,
        only the score is returned.
    """
    if len(seq1) != len(seq2):
        raise ValueError(
            f"Different sequence lengths ({len(seq1):d} and {len(seq2):d})"
        )
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    score = _add_scores(seq1.code, seq2.code, matrix.score_matrix())
    if score_only:
        return score
    else:
        # Sequences do not need to be actually aligned
        # -> Create alignment with trivial trace
        # [[0 0]
        #  [1 1]
        #  [2 2]
        #   ... ]
        seq_length = len(seq1)
        return Alignment(
            sequences = [seq1, seq2],
            trace     = np.tile(np.arange(seq_length), 2)
                        .reshape(2, seq_length)
                        .transpose(),
            score     = score
        )


@cython.boundscheck(False)
@cython.wraparound(False)
def _add_scores(CodeType1[:] code1 not None,
                CodeType2[:] code2 not None,
                const int32[:,:] matrix not None):
    cdef int32 score = 0
    cdef int i
    for i in range(code1.shape[0]):
        score += matrix[code1[i], code2[i]]
    return score


def align_optimal(seq1, seq2, matrix, gap_penalty=-10,
                  terminal_penalty=True, local=False,
                  max_number=1000):
    """
    align_optimal(seq1, seq2, matrix, gap_penalty=-10,
                  terminal_penalty=True, local=False, max_number=1000)

    Perform an optimal alignment of two sequences based on a
    dynamic programming algorithm.

    This algorithm yields an optimal alignment, i.e. the sequences
    are aligned in the way that results in the highest similarity
    score. This operation can be very time and space consuming,
    because both scale linearly with each sequence length.

    The aligned sequences do not need to be instances from the same
    :class:`Sequence` subclass, since they do not need to have the same
    alphabet. The only requirement is that the
    :class:`SubstitutionMatrix`' alphabets extend the alphabets of the
    two sequences.

    This function can either perform a global alignment, based on the
    Needleman-Wunsch algorithm :footcite:`Needleman1970` or a local
    alignment, based on the Smith–Waterman algorithm
    :footcite:`Smith1981`.

    Furthermore this function supports affine gap penalties using the
    Gotoh algorithm :footcite:`Gotoh1982`, however, this requires
    approximately 4 times the RAM space and execution time.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        linear gap penalty.
        If a tuple is provided, an affine gap penalty is used.
        The first integer in the tuple is the gap opening penalty,
        the second integer is the gap extension penalty.
        The values need to be negative.
    terminal_penalty : bool, optional
        If true, gap penalties are applied to terminal gaps.
        If `local` is true, this parameter has no effect.
    local : bool, optional
        If false, a global alignment is performed, otherwise a local
        alignment is performed.
    max_number : int, optional
        The maximum number of alignments returned.
        When the number of branches exceeds this value in the traceback
        step, no further branches are created.

    Returns
    -------
    alignments : list, type=Alignment
        A list of alignments.
        Each alignment in the list has the same maximum similarity
        score.

    See Also
    --------
    align_banded

    References
    ----------

    .. footbibliography::

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


    # This implementation uses transposed tables in comparison
    # to the common visualization
    # This means the first sequence is one the left
    # and the second sequence is at the top
    trace_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.uint8)
    code1 = seq1.code
    code2 = seq2.code

    # Table filling
    ###############
    if affine_penalty:
        # Affine gap penalty
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
        # Value for negative infinity
        # Used to prevent unallowed state transitions
        # Subtraction of gap_open, gap_ext and lowest score value
        # to prevent integer overflow
        neg_inf = np.iinfo(np.int32).min - gap_open - gap_ext
        min_score = np.min(matrix.score_matrix())
        if min_score < 0:
            neg_inf -= min_score
        # m_table, g1_table and g2_table are the 3 score tables
        m_table  = np.zeros((len(seq1)+1, len(seq2)+1), dtype=np.int32)
        # Fill with negative infinity values to prevent that an
        # alignment trace starts with a gap extension
        # instead of a gap opening
        g1_table = np.full((len(seq1)+1, len(seq2)+1), neg_inf, dtype=np.int32)
        g2_table = np.full((len(seq1)+1, len(seq2)+1), neg_inf, dtype=np.int32)
        # Disallow trace coming from the match table on the
        # left column/top row, as these represent terminal gaps
        m_table [0, 1:] = neg_inf
        m_table [1:, 0] = neg_inf
        # Initialize first row and column for global alignments
        if not local:
            if terminal_penalty:
                # Terminal gaps are penalized
                # -> Penalties in first row/column
                g1_table[0, 1:] = (np.arange(len(seq2)) * gap_ext) + gap_open
                g2_table[1:, 0] = (np.arange(len(seq1)) * gap_ext) + gap_open
            else:
                g1_table[0, 1:] = np.zeros(len(seq2))
                g2_table[1:, 0] = np.zeros(len(seq1))
            trace_table[0,  1] = TraceDirectionAffine.MATCH_TO_GAP_LEFT
            trace_table[0, 2:] = TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
            trace_table[1,  0] = TraceDirectionAffine.MATCH_TO_GAP_TOP
            trace_table[2: ,0] = TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
        else:
            g1_table[0, 1:] = np.zeros(len(seq2))
            g2_table[1:, 0] = np.zeros(len(seq1))
        _fill_align_table_affine(code1, code2,
                                 matrix.score_matrix(), trace_table,
                                 m_table, g1_table, g2_table,
                                 gap_open, gap_ext, terminal_penalty, local)
    else:
        # Linear gap penalty
        # The table for saving the scores
        score_table = np.zeros(( len(seq1)+1, len(seq2)+1 ), dtype=np.int32)
        # Initialize first row and column for global alignments
        if not local:
            if terminal_penalty:
                # Terminal gaps are penalized
                # -> Penalties in first row/column
                score_table[:,0] = np.arange(len(seq1)+1) * gap_penalty
                score_table[0,:] = np.arange(len(seq2)+1) * gap_penalty
            trace_table[1:,0] = TraceDirectionLinear.GAP_TOP
            trace_table[0,1:] = TraceDirectionLinear.GAP_LEFT
        _fill_align_table(code1, code2, matrix.score_matrix(), trace_table,
                          score_table, gap_penalty, terminal_penalty, local)


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
    if local:
        # The start point is the maximal score in the table
        # Multiple starting points possible,
        # when duplicates of maximal score exist
        if affine_penalty:
            # The maximum score in the gap score tables do not need to
            # be considered, as these starting positions would indicate
            # that the local alignment starts with a gap
            # Hence the maximum score value in these tables is always
            # less than in the match table
            max_score = np.max(m_table)
            i_list, j_list = np.where((m_table == max_score))
            state_list = np.append(state_list, np.full(len(i_list), 1))
        else:
            max_score = np.max(score_table)
            i_list, j_list = np.where((score_table == max_score))
            # State is always 0 for linear gap penalty
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
    return [Alignment([seq1, seq2], trace, max_score) for trace in trace_list]


@cython.boundscheck(False)
@cython.wraparound(False)
def _fill_align_table(CodeType1[:] code1 not None,
                      CodeType2[:] code2 not None,
                      const int32[:,:] matrix not None,
                      uint8[:,:] trace_table not None,
                      int32[:,:] score_table not None,
                      int gap_penalty,
                      bint term_penalty,
                      bint local):
    """
    Fill an alignment table with linear gap penalty using dynamic
    programming.

    Parameters
    ----------
    code1, code2
        The sequence code of each sequence to be aligned.
    matrix
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
    cdef int max_i, max_j
    cdef int32 from_diag, from_left, from_top
    cdef uint8 trace
    cdef int32 score

    # For local alignments terminal gaps on the right side are ignored
    # anyway, as the alignment should stop before
    if local:
        term_penalty = True
    # Used in case terminal gaps are not penalized
    i_max = score_table.shape[0] -1
    j_max = score_table.shape[1] -1

    # Starts at 1 since the first row and column are already filled
    for i in range(1, score_table.shape[0]):
        for j in range(1, score_table.shape[1]):
            # Evaluate score from diagonal direction
            # -1 in sequence index is necessary
            # due to the shift of the sequences
            # to the bottom/right in the table
            from_diag = score_table[i-1, j-1] + matrix[code1[i-1], code2[j-1]]
            # Evaluate score from left direction
            if not term_penalty and i == i_max:
                from_left = score_table[i, j-1]
            else:
                from_left = score_table[i, j-1] + gap_penalty
            # Evaluate score from top direction
            if not term_penalty and j == j_max:
                from_top = score_table[i-1, j]
            else:
                from_top = score_table[i-1, j] + gap_penalty

            trace = get_trace_linear(from_diag, from_left, from_top, &score)

            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then the score of the cell remains 0
            # and the trace ends here
            if local == True and score <= 0:
                continue

            score_table[i,j] = score
            trace_table[i,j] = trace


@cython.boundscheck(False)
@cython.wraparound(False)
def _fill_align_table_affine(CodeType1[:] code1 not None,
                             CodeType2[:] code2 not None,
                             const int32[:,:] matrix not None,
                             uint8[:,:] trace_table not None,
                             int32[:,:] m_table not None,
                             int32[:,:] g1_table not None,
                             int32[:,:] g2_table not None,
                             int gap_open,
                             int gap_ext,
                             bint term_penalty,
                             bint local):
    """
    Fill an alignment table with affine gap penalty using dynamic
    programming.

    Parameters
    ----------
    code1, code2
        The sequence code of each sequence to be aligned.
    matrix
        The score matrix obtained from the class:`SubstitutionMatrix`
        object.
    trace_table
        A matrix containing values indicating the direction for the
        traceback step.
        The matrix is filled in this function.
    m_table, g1_table, g2_table
        The alignment tables containing the scores.
        `m_table` contains values for matches.
        `g1_table` contains values for gaps in the first sequence.
        `g2_table` contains values for gaps in the second sequence.
        The matrix is filled in this function.
    gap_open
        The gap opening penalty.
    gap_ext
        The gap extension penalty.
    term_penalty
        Indicates, whether terminal gaps should be penalized.
    local
        Indicates, whether a local alignment should be performed.
    """

    cdef int i, j
    cdef int max_i, max_j
    cdef int32 mm_score, g1m_score, g2m_score
    cdef int32 mg1_score, g1g1_score
    cdef int32 mg2_score, g2g2_score
    cdef int32 m_score, g1_score, g2_score
    cdef int32 similarity_score
    cdef uint8 trace

    # For local alignments terminal gaps on the right and the bottom are
    # ignored anyway, as the alignment should stop before
    if local:
        term_penalty = True
    # Used in case terminal gaps are not penalized
    i_max = trace_table.shape[0] -1
    j_max = trace_table.shape[1] -1

    # Starts at 1 since the first row and column are already filled
    for i in range(1, trace_table.shape[0]):
        for j in range(1, trace_table.shape[1]):
            # Calculate the scores for possible transitions
            # into the current cell
            similarity_score = matrix[code1[i-1], code2[j-1]]
            mm_score  =  m_table[i-1,j-1] + similarity_score
            g1m_score = g1_table[i-1,j-1] + similarity_score
            g2m_score = g2_table[i-1,j-1] + similarity_score
            # No transition from g1_table to g2_table and vice versa
            # Since this would mean adjacent gaps in both sequences
            # A substitution makes more sense in this case
            if not term_penalty and i == i_max:
                mg1_score  =  m_table[i,j-1]
                g1g1_score = g1_table[i,j-1]
            else:
                mg1_score  =  m_table[i,j-1] + gap_open
                g1g1_score = g1_table[i,j-1] + gap_ext
            if not term_penalty and j == j_max:
                mg2_score  = m_table[i-1,j]
                g2g2_score = g2_table[i-1,j]
            else:
                mg2_score  =  m_table[i-1,j] + gap_open
                g2g2_score = g2_table[i-1,j] + gap_ext

            trace = get_trace_affine(
                mm_score, g1m_score, g2m_score,
                mg1_score, g1g1_score,
                mg2_score, g2g2_score,
                # The max score values to be written
                &m_score, &g1_score, &g2_score
            )

            # Fill values into tables
            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then the score of the cell remains 0
            # and the trace ends here
            if local == True:
                if m_score <= 0:
                    # End trace in specific table
                    # by filtering out the respective bits
                    trace &= ~(
                        TraceDirectionAffine.MATCH_TO_MATCH |
                        TraceDirectionAffine.GAP_LEFT_TO_MATCH |
                        TraceDirectionAffine.GAP_TOP_TO_MATCH
                    )
                    # m_table[i,j] remains 0
                else:
                    m_table[i,j] = m_score
                if g1_score <= 0:
                    trace &= ~(
                        TraceDirectionAffine.MATCH_TO_GAP_LEFT |
                        TraceDirectionAffine.GAP_LEFT_TO_GAP_LEFT
                    )
                    # g1_table[i,j] remains negative infinity
                else:
                    g1_table[i,j] = g1_score
                if g2_score <= 0:
                    trace &= ~(
                        TraceDirectionAffine.MATCH_TO_GAP_TOP |
                        TraceDirectionAffine.GAP_TOP_TO_GAP_TOP
                    )
                    # g2_table[i,j] remains negative infinity
                else:
                    g2_table[i,j] = g2_score
            else:
                m_table[i,j] = m_score
                g1_table[i,j] = g1_score
                g2_table[i,j] = g2_score
            trace_table[i,j] = trace