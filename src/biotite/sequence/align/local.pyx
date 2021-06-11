# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_local_ungapped"]

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
    uint16
    uint32
    uint64
ctypedef fused CodeType2:
    uint8
    uint16
    uint32
    uint64


def align_local_ungapped(seq1, seq2, matrix, seed, int32 threshold,
                         str direction="both", bint score_only=False):
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

    >>> TODO
    """
    if     not matrix.get_alphabet1().extends(seq1.get_alphabet()) \
        or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
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

    # Range check to avoid negative indices
    if upstream and seq1_start > 0 and seq2_start > 0:
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
    cdef int i
    cdef int32 score = 0, max_score = 0
    cdef int i_max_score = -1

    cdef int min_length
    if code1.shape[0] < code2.shape[0]:
        min_length = code1.shape[0]
    else:
        min_length = code2.shape[0]

    for i in range(min_length):
        score += matrix[code1[i], code2[i]]
        print(score, threshold)
        if score >= max_score:
            max_score = score
            i_max_score = i
        elif i_max_score - score > threshold:
            break

    return max_score, i_max_score + 1