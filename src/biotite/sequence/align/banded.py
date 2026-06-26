# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_banded"]

from typing import TYPE_CHECKING, Literal, overload
import numpy as np
from biotite.rust.sequence.align import align_banded as _rust_align_banded
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.align.pairwise import prepare_scoring

if TYPE_CHECKING:
    from biotite.sequence.sequence import Sequence
    from biotite.typing import S1, S2


@overload
def align_banded(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    band: tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    local: bool = False,
    max_number: int = 1000,
    score_only: Literal[False] = False,
) -> list[Alignment]: ...
@overload
def align_banded(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    band: tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    local: bool = False,
    max_number: int = 1000,
    *,
    score_only: Literal[True],
) -> int: ...
def align_banded(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix | tuple[int, int],
    band: tuple[int, int],
    gap_penalty: int | tuple[int, int] = -10,
    local: bool = False,
    max_number: int = 1000,
    score_only: bool = False,
) -> list[Alignment] | int:
    r"""
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
    matrix : SubstitutionMatrix or tuple(int, int)
        Either a substitution matrix or a ``(match, mismatch)`` pair of scores.
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
    score_only : bool, optional
        If true, only the similarity score is returned instead of the
        list of alignments.

    Returns
    -------
    alignments : list of Alignment or int
        The generated alignments.
        Each alignment in the list has the same similarity score,
        which is the maximum score possible within the defined band.
        If `score_only` is set to true, only the score is returned.

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

    # The shorter sequence is the one on the left of the matrix
    # -> shorter sequence is 'seq1'
    if len(seq2) < len(seq1):
        seq1, seq2 = seq2, seq1
        band = (-band[0], -band[1])
        if isinstance(matrix, SubstitutionMatrix):
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
    lower_diag = max(lower_diag, -len(seq1) + 1)
    upper_diag = min(upper_diag, len(seq2) - 1)
    if upper_diag - lower_diag + 1 < 1:
        raise ValueError("The width of the band is 0")

    # The two sequences may use different code dtypes
    # -> widen both to a common type so the Rust layer dispatches once
    common_dtype = np.promote_types(seq1.code.dtype, seq2.code.dtype)

    traces, score = _rust_align_banded(
        np.ascontiguousarray(seq1.code, common_dtype),
        np.ascontiguousarray(seq2.code, common_dtype),
        prepare_scoring(matrix, seq1, seq2),
        gap_penalty,
        lower_diag,
        upper_diag,
        local,
        score_only,
        max_number,
    )

    if score_only:
        return score
    if is_swapped:
        return [
            Alignment([seq2, seq1], np.flip(trace, axis=1), score) for trace in traces
        ]
    else:
        return [Alignment([seq1, seq2], trace, score) for trace in traces]
