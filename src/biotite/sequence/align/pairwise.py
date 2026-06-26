# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_ungapped", "align_optimal"]

from typing import TYPE_CHECKING, Literal, overload
import numpy as np
from biotite.rust.sequence.align import align_optimal as _rust_align_optimal
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix

if TYPE_CHECKING:
    from biotite.sequence.sequence import Sequence
    from biotite.typing import S1, S2


@overload
def align_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    score_only: Literal[False] = False,
) -> Alignment: ...
@overload
def align_ungapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    score_only: Literal[True],
) -> int: ...
def align_ungapped(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix | tuple[int, int],
    score_only: bool = False,
) -> Alignment | int:
    """
    Align two sequences without insertion of gaps.

    Both sequences need to have the same length.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences, whose similarity should be scored.
    matrix : SubstitutionMatrix or tuple(int, int)
        Either a substitution matrix or a ``(match, mismatch)`` pair of scores.
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
    if isinstance(matrix, SubstitutionMatrix):
        if not matrix.get_alphabet1().extends(
            seq1.get_alphabet()
        ) or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
        # Sum the substitution scores of the aligned symbols via fancy indexing
        score = int(matrix.score_matrix()[seq1.code, seq2.code].sum())
    else:
        # Match/mismatch scoring
        match_score, mismatch_score = matrix
        score = int(np.where(seq1.code == seq2.code, match_score, mismatch_score).sum())
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
            sequences=[seq1, seq2],
            trace=np.tile(np.arange(seq_length), 2).reshape(2, seq_length).transpose(),
            score=score,
        )


@overload
def align_optimal(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
    local: bool = False,
    max_number: int = 1000,
    score_only: Literal[False] = False,
) -> list[Alignment]: ...
@overload
def align_optimal(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
    local: bool = False,
    max_number: int = 1000,
    *,
    score_only: Literal[True],
) -> int: ...
def align_optimal(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix | tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    terminal_penalty: bool = True,
    local: bool = False,
    max_number: int = 1000,
    score_only: bool = False,
) -> list[Alignment] | int:
    """
    Perform an optimal alignment of two sequences based on a
    dynamic programming algorithm.

    This algorithm yields an optimal alignment, i.e. the sequences
    are aligned in the way that results in the highest similarity
    score. This operation can be very time and space consuming,
    because both scale linearly with each sequence length.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix or tuple(int, int)
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
    score_only : bool, optional
        If true, only the similarity score is returned instead of the
        list of alignments.

    Returns
    -------
    alignments : list, type=Alignment or int
        A list of alignments.
        Each alignment in the list has the same maximum similarity
        score.
        If `score_only` is set to true, only the score is returned.

    See Also
    --------
    align_banded
        For a faster heuristic alignment constrained to a diagonal band.

    Notes
    -----
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
    # Check if gap penalty is linear or affine
    if isinstance(gap_penalty, int):
        if gap_penalty > 0:
            raise ValueError("Gap penalty must be negative")
    elif isinstance(gap_penalty, tuple):
        if gap_penalty[0] > 0 or gap_penalty[1] > 0:
            raise ValueError("Gap penalty must be negative")
    else:
        raise TypeError("Gap penalty must be either integer or tuple")
    # Check if max_number is reasonable
    if max_number < 1:
        raise ValueError("Maximum number of returned alignments must be at least 1")

    # The two sequences may use different code dtypes
    # -> widen both to a common type so the Rust layer dispatches once
    common_dtype = np.promote_types(seq1.code.dtype, seq2.code.dtype)

    traces, score = _rust_align_optimal(
        np.ascontiguousarray(seq1.code, common_dtype),
        np.ascontiguousarray(seq2.code, common_dtype),
        prepare_scoring(matrix, seq1, seq2),
        gap_penalty,
        terminal_penalty,
        local,
        score_only,
        max_number,
    )

    if score_only:
        return score
    return [Alignment([seq1, seq2], trace, score) for trace in traces]


def prepare_scoring(
    scoring: SubstitutionMatrix | tuple[int, int],
    seq1: Sequence | None,
    seq2: Sequence | None,
    check_matrix: bool = True,
) -> np.ndarray | tuple[int, int]:
    """
    Resolve the `scoring` argument of the alignment functions into the value
    handed to the Rust layer.

    `scoring` is either a :class:`SubstitutionMatrix` (whose 2D score matrix is
    returned) or a ``(match, mismatch)`` tuple of scores (returned as a tuple of
    plain integers). In the matrix case the matrix alphabets are checked against
    the sequences, unless `check_matrix` is false.

    This helper is shared across the alignment modules, but is intentionally
    kept out of ``__all__`` as it is not part of the public API.

    Parameters
    ----------
    scoring : SubstitutionMatrix or tuple(int, int)
        The scoring scheme to resolve.
    seq1, seq2 : Sequence or None
        The sequences to check the matrix alphabets against.
        May be ``None`` if `check_matrix` is false.
    check_matrix : bool, optional
        Whether to check the matrix alphabets against the sequences.

    Returns
    -------
    scoring : ndarray or tuple(int, int)
        The 2D score matrix or the ``(match, mismatch)`` tuple.
    """
    if isinstance(scoring, tuple):
        match_score, mismatch_score = scoring
        return (int(match_score), int(mismatch_score))
    if check_matrix:
        # If the matrix is checked, the sequences are required
        if seq1 is None or seq2 is None:
            raise TypeError("Sequences are required to check the matrix")
        if not scoring.get_alphabet1().extends(
            seq1.get_alphabet()
        ) or not scoring.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    return scoring.score_matrix()
