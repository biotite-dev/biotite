# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_banded"]

cimport cython
cimport numpy as np
from .tracetable cimport follow_trace, get_trace_linear, get_trace_affine, \
                         TraceDirectionAffine, TraceState

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
    uint16
    uint32
    uint64
ctypedef fused CodeType2:
    uint8
    uint16
    uint32
    uint64


def align_banded(seq1, seq2, matrix, band, gap_penalty=-10, local=False,
                 max_number=1000):
    """
    align_banded(seq1, seq2, matrix, band, gap_penalty=-10, local=False,
                 max_number=1000)

    Perform a local or semi-global alignment within a defined diagonal
    band. :footcite:`Pearson1988`

    The function requires two diagonals that defines the lower
    and upper limit of the alignment band.
    A diagonal is an integer defined as :math:`D = j - i`, where *i* and
    *j* are sequence positions in the first and second sequence,
    respectively.
    This means that two symbols at position *i* and *j* can only be
    aligned to each other, if :math:`D_L \leq j - i \leq D_U`.
    With increasing width of the diagonal band, the probability to find
    the optimal alignment, but also the computation time increases.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix
        The substitution matrix used for scoring.
    band : tuple(int, int)
        The diagonals that represent the lower and upper limit of the
        search space.
        A diagonal :math:`D` is defined as :math:`D = j-i`, where
        :math:`i` and :math:`j` are positions in `seq1` and `seq2`,
        respectively.
        An alignment of sequence positions where :math:`D` is lower than
        the lower limit or greater than the upper limit is not explored
        by the algorithm.
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        linear gap penalty.
        If a tuple is provided, an affine gap penalty is used.
        The first integer in the tuple is the gap opening penalty,
        the second integer is the gap extension penalty.
        The values need to be negative.
    local : bool, optional
        If set to true, a local alignment is performed.
        Otherwise (default) a semi-global alignment is performed.
    max_number : int, optional
        The maximum number of alignments returned.
        When the number of branches exceeds this value in the traceback
        step, no further branches are created.

    Returns
    -------
    alignments : list of Alignment
        The generated alignments.
        Each alignment in the list has the same similarity score,
        which is the maximum score possible within the defined band.

    See Also
    --------
    align_optimal
        Guarantees to find the optimal alignment at the cost of greater
        compuation time and memory requirements.

    Notes
    -----
    The diagonals give the maximum difference between the
    number of inserted gaps.
    This means for any position in the alignment, the algorithm
    will not consider inserting a gap into a sequence, if the first
    sequence has already ``-band[0]`` more gaps than the second
    sequence or if the second sequence has already ``band[1]`` more gaps
    than the first sequence, even if inserting additional gaps would
    yield a more optimal alignment.
    Considerations on how to find a suitable band width are discussed in
    :footcite:`Gibrat2018`.

    The restriction to a limited band is the central difference between
    the banded alignment heuristic and the optimal alignment
    algorithms :footcite:`Needleman1970, Smith1981`.
    Those classical algorithms require :math:`O(m \cdot n)`
    memory space and computation time for aligning two sequences with
    lengths :math:`m` and :math:`n`, respectively.
    The banded alignment algorithm reduces both requirements to
    :math:`O(\min(m,n) \cdot (D_U - D_L))`.

    *Implementation details*

    The implementation is very similar to :func:`align_optimal()`.
    The most significant difference is that not the complete alignment
    table is filled, but only the cells that lie within the diagonal
    band.
    Furthermore, to reduce also the space requirements the diagnoal band
    is 'straightened', i.e. the table's rows are indented to the left.
    Hence, this table

    = = = = = = = = = =
    . . x x x . . . . .
    . . . x x x . . . .
    . . . . x x x . . .
    . . . . . x x x . .
    . . . . . . x x x .
    = = = = = = = = = =

    is transformed into this table:

    = = =
    x x x
    x x x
    x x x
    x x x
    x x x
    = = =

    Filled cells, i.e. cells within the band, are indicated by ``x``.
    The shorter sequence is always represented by the first dimension
    of the table in this implementation.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    Find a matching diagonal for two sequences:

    >>> sequence1 = NucleotideSequence("GCGCGCTATATTATGCGCGC")
    >>> sequence2 = NucleotideSequence("TATAAT")
    >>> table = KmerTable.from_sequences(k=4, sequences=[sequence1])
    >>> match = table.match(sequence2)[0]
    >>> diagonal = match[0] - match[2]
    >>> print(diagonal)
    -6

    Align the sequences centered on the diagonal with buffer in both
    directions:

    >>> BUFFER = 5
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> alignment = align_banded(
    ...     sequence1, sequence2, matrix,
    ...     band=(diagonal - BUFFER, diagonal + BUFFER), gap_penalty=(-6, -1)
    ... )[0]
    >>> print(alignment)
    TATATTAT
    TATA--AT
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
    if len(seq1) + upper_diag <= 0 or lower_diag >= len(seq2):
        raise ValueError(
            "Alignment band is out of range, the band allows no overlap "
            "between both sequences"
        )
    # Crop band diagonals to reasonable size, so that it at maximum
    # covers the search space of an unbanded alignment
    lower_diag = max(lower_diag, -len(seq1)+1)
    upper_diag = min(upper_diag,  len(seq2)-1)
    band_width = upper_diag - lower_diag + 1
    if band_width < 1:
        raise ValueError("The width of the band is 0")

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
    # dynamic programming matrix should not be used, since it would be
    # outside the band
    # It is the 'worst' score available, so the trace table will never
    # include such a direction
    neg_inf = np.iinfo(np.int32).min
    # Correct the 'negative infinity' integer, by making it more positive
    # This prevents an integer underflow when the gap penalty or
    # match score is added to this value
    neg_inf -= min(gap_penalty) if affine_penalty else gap_penalty
    min_score = np.min(matrix.score_matrix())
    if min_score < 0:
        neg_inf -= min_score

    if affine_penalty:
        # Affine gap penalty
        gap_open = gap_penalty[0]
        gap_ext = gap_penalty[1]
        # m_table, g1_table and g2_table are the 3 score tables
        m_table = np.zeros((len(seq1)+1, band_width+2), dtype=np.int32)
        # Fill with negative infinity values to prevent that an
        # alignment trace starts with a gap extension
        # instead of a gap opening
        g1_table = np.full((len(seq1)+1, band_width+2), neg_inf, np.int32)
        g2_table = np.full((len(seq1)+1, band_width+2), neg_inf, np.int32)
        # As explained for the trace table (see above),
        # the score table is filled with with netagive infinty values
        # on the left and right column to prevent the trace leaving the
        # alignment band
        m_table[:,  0] = neg_inf
        m_table[:, -1] = neg_inf
        # Initialize first row and column for global alignments
        _fill_align_table_affine(code1, code2,
                                 matrix.score_matrix(), trace_table,
                                 m_table, g1_table, g2_table,
                                 lower_diag, upper_diag,
                                 gap_open, gap_ext, local)
    else:
        # Linear gap penalty
        score_table = np.zeros((len(seq1)+1, band_width+2), dtype=np.int32)
        score_table[:,  0] = neg_inf
        score_table[:, -1] = neg_inf
        _fill_align_table(
            code1, code2, matrix.score_matrix(), trace_table, score_table,
            lower_diag, upper_diag, gap_penalty, local
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
    # `state_list` lists of start states
    # State specifies the table the trace starts in
    if local:
        # The start point is the maximal score in the table
        # Multiple starting points possible,
        # when duplicates of maximum score exist
        if affine_penalty:
            # The maximum score in the gap score tables do not need to
            # be considered, as these starting positions would indicate
            # that the local alignment starts with a gap
            # Hence the maximum score value in these tables is always
            # less than in the match table
            max_score = np.max(m_table)
            i_list, j_list = np.where((m_table == max_score))
            state_list = np.full(
                len(i_list), TraceState.MATCH_STATE, dtype=int
            )
        else:
            max_score = np.max(score_table)
            i_list, j_list = np.where((score_table == max_score))
            # State is always 0 for linear gap penalty
            # since there is only one table
            state_list = np.full(
                len(i_list), TraceState.NO_STATE, dtype=int
            )
    else:
        # Get all allowed trace start indices
        possible_i_start, possible_j_start = get_global_trace_starts(
            len(seq1), len(seq2), lower_diag, upper_diag
        )
        if affine_penalty:
            state_list = np.zeros(0, dtype=int)
            m_scores  =  m_table[possible_i_start, possible_j_start]
            g1_scores = g1_table[possible_i_start, possible_j_start]
            g2_scores = g2_table[possible_i_start, possible_j_start]
            m_max_score  = np.max(m_scores)
            g1_max_score = np.max(g1_scores)
            g2_max_score = np.max(g2_scores)
            max_score = max(m_max_score, g1_max_score, g2_max_score)
            if m_max_score == max_score:
                best_indices = np.where(m_scores == max_score)[0]
                i_list = np.append(i_list, possible_i_start[best_indices])
                j_list = np.append(j_list, possible_j_start[best_indices])
                state_list = np.append(
                    state_list,
                    np.full(len(best_indices),
                    TraceState.MATCH_STATE, dtype=int)
                )
            if g1_max_score == max_score:
                best_indices = np.where(g1_scores == max_score)[0]
                i_list = np.append(i_list, possible_i_start[best_indices])
                j_list = np.append(j_list, possible_j_start[best_indices])
                state_list = np.append(
                    state_list,
                    np.full(len(best_indices),
                    TraceState.GAP_LEFT_STATE, dtype=int)
                )
            if g2_max_score == max_score:
                best_indices = np.where(g2_scores == max_score)[0]
                i_list = np.append(i_list, possible_i_start[best_indices])
                j_list = np.append(j_list, possible_j_start[best_indices])
                state_list = np.append(
                    state_list,
                    np.full(len(best_indices),
                    TraceState.GAP_TOP_STATE, dtype=int)
                )
        else:
            # Choose the trace start index with the highest score
            # in the score table
            scores = score_table[possible_i_start, possible_j_start]
            max_score = np.max(scores)
            best_indices = np.where(scores == max_score)
            i_list = possible_i_start[best_indices]
            j_list = possible_j_start[best_indices]
            state_list = np.full(
                len(i_list), TraceState.NO_STATE, dtype=int
            )

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
        follow_trace(
            trace_table, True, i_start, j_start, 0,
            trace, trace_list, state=state_start,
            curr_trace_count=&curr_trace_count, max_trace_count=max_number,
            lower_diag=lower_diag, upper_diag=upper_diag
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
    # 'follow_trace()', however, if multiple alignment starts
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
    local
        Indicates, whether a local alignment should be performed.
    """

    cdef int i, j
    cdef int seq_i, seq_j
    cdef int32 from_diag, from_left, from_top
    cdef uint8 trace
    cdef int32 score

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

            trace = get_trace_linear(from_diag, from_left, from_top, &score)

            # Local alignment specialty:
            # If score is less than or equal to 0,
            # then 0 is saved on the field and the trace ends here
            if local == True and score <= 0:
                score_table[i,j] = 0
            else:
                score_table[i,j] = score
                trace_table[i,j] = trace


@cython.boundscheck(False)
@cython.wraparound(False)
def _fill_align_table_affine(CodeType1[:] code1 not None,
                             CodeType2[:] code2 not None,
                             const int32[:,:] mat not None,
                             uint8[:,:] trace_table not None,
                             int32[:,:] m_table not None,
                             int32[:,:] g1_table not None,
                             int32[:,:] g2_table not None,
                             int lower_diag,
                             int upper_diag,
                             int gap_open,
                             int gap_ext,
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
    local
        Indicates, whether a local alignment should be performed.
    """

    cdef int i, j
    cdef int seq_i, seq_j
    cdef int32 mm_score, g1m_score, g2m_score
    cdef int32 mg1_score, g1g1_score
    cdef int32 mg2_score, g2g2_score
    cdef uint8 trace
    cdef int32 m_score, g1_score, g2_score
    cdef int32 similarity_score

    # Starts at 1 since the first row and column are already fil
    for seq_i in range(0, code1.shape[0]):
        i = seq_i + 1
        for seq_j in range(
            max(0,              seq_i + lower_diag),
            min(code2.shape[0], seq_i + upper_diag+1)
        ):
            j = seq_j - seq_i - lower_diag + 1
            # Calculate the scores for possible transitions
            # into the current cell
            similarity_score = mat[code1[seq_i], code2[seq_j]]
            mm_score  =  m_table[i-1, j] + similarity_score
            g1m_score = g1_table[i-1, j] + similarity_score
            g2m_score = g2_table[i-1, j] + similarity_score
            # No transition from g1_table to g2_table and vice versa
            # Since this would mean adjacent gaps in both sequences
            # A substitution makes more sense in this case
            mg1_score  =  m_table[i, j-1] + gap_open
            g1g1_score = g1_table[i, j-1] + gap_ext
            mg2_score  =  m_table[i-1, j+1] + gap_open
            g2g2_score = g2_table[i-1, j+1] + gap_ext

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