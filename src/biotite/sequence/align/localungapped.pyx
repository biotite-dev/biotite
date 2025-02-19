# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_local_ungapped"]

cimport cython
cimport numpy as np

import numpy as np
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


def align_local_ungapped(seq1, seq2, matrix, seed, int32 threshold,
                         str direction="both", bint score_only=False,
                         bint check_matrix=True):
    """
    align_local_ungapped(seq1, seq2, matrix, seed, threshold,
                         direction="both", score_only=False, check_matrix=True)

    Perform a local alignment extending from given `seed` position
    without inserting gaps.

    The alignment extends into one or both directions (controlled by
    `direction`) until the total alignment score falls more than
    `threshold` below the maximum score found (*X-Drop*).
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
    check_matrix : bool, optional
        If set to False, the `matrix` is not checked for compatibility
        with the alphabets of the sequences.
        Due to the small overall runtime of the function, this can increase
        performance substantially.
        However, unexpected results or crashes may occur, if an
        incompatible `matrix` is given.


    Returns
    -------
    alignment : Alignment
        The resulting ungapped alignment.
        Only returned, if `score_only` is ``False``.
    score : int
        The alignment similarity score.
        Only returned, if `score_only` is ``True``.

    See Also
    --------
    align_gapped
        For gapped local alignments with the same *X-Drop* technique.

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
    if seq1_start < 0 or seq2_start < 0:
        raise IndexError("Seed must contain positive indices")

    cdef np.ndarray code1 = seq1.code
    cdef np.ndarray code2 = seq2.code
    # For C- function call of the '_seed_extend_uint8()' function
    # for the common case
    # This gives significant performance increase since the
    # seed extend itself runs fast
    cdef bint both_uint8 = (code1.dtype == np.uint8) \
                         & (code2.dtype == np.uint8)

    cdef int32 length
    cdef int start_offset = 0
    cdef int stop_offset = 1
    cdef int32 score = 0
    cdef int32 total_score = 0

    # Separate alignment into two parts:
    # the regions upstream and downstream from the seed position
    # Range check to avoid negative indices
    if upstream and seq1_start > 0 and seq2_start > 0:
        # For the upstream region the respective part of the sequence
        # must be reversed
        if both_uint8:
            length = _seed_extend_uint8(
                code1[seq1_start-1::-1], code2[seq2_start-1::-1],
                score_matrix, threshold, &score
            )
        else:
            score, length = _seed_extend_generic(
                code1[seq1_start-1::-1], code2[seq2_start-1::-1],
                score_matrix, threshold
            )
        total_score += score
        start_offset -= length
    if downstream:
        if both_uint8:
            length = _seed_extend_uint8(
                code1[seq1_start+1:], code2[seq2_start+1:],
                score_matrix, threshold, &score
            )
        else:
            score, length = _seed_extend_generic(
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
def _seed_extend_generic(CodeType1[:] code1 not None,
                         CodeType2[:] code2 not None,
                         const int32[:,:] matrix not None,
                         int32 threshold):
    """
    Align two sequences without insertion of gaps beginning from
    start of the given sequences.
    If the score drops too low, terminate the alignment.
    Return the similarity score and the number of aligned symbols.
    """
    cdef int i
    cdef int32 total_score = 0, max_score = 0
    cdef int i_max_score = -1

    # Iterate over the symbols in both sequences
    # The alignment automatically terminates,
    # if the the end of either sequence is reached
    for i in range(_min(code1.shape[0], code2.shape[0])):
        total_score += matrix[code1[i], code2[i]]
        if total_score >= max_score:
            max_score = total_score
            i_max_score = i
        elif max_score - total_score > threshold:
            # Score drops too low -> terminate alignment
            break

    # Return the total score and the number of aligned symbols at the
    # point with maximum total score
    return max_score, i_max_score + 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _seed_extend_uint8(uint8[:] code1, uint8[:] code2,
                            const int32[:,:] matrix,
                            int32 threshold, int32* score):
    """
    The same functionality as :func:`_seed_extend_generic()` but as
    C-function tailored for the common ``uint8`` sequence code *dtype*.
    This increases the performance for this common case.
    """
    cdef int i
    cdef int32 total_score = 0, max_score = 0
    cdef int i_max_score = -1

    # Iterate over the symbols in both sequences
    # The alignment automatically terminates,
    # if the the end of either sequence is reached
    for i in range(_min(code1.shape[0], code2.shape[0])):
        total_score += matrix[code1[i], code2[i]]
        if total_score >= max_score:
            max_score = total_score
            i_max_score = i
        elif max_score - total_score > threshold:
            # Score drops too low -> terminate alignment
            break

    # Return the total score and the number of aligned symbols at the
    # point with maximum total score
    score[0] = max_score
    return i_max_score + 1


cdef inline int _min(int a, int b):
    return a if a < b else b