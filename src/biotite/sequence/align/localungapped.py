# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["SeedExtension", "align_local_ungapped"]

import warnings
from typing import TYPE_CHECKING
from biotite.rust.sequence.align import SeedExtension

if TYPE_CHECKING:
    from biotite.sequence.align.alignment import Alignment
    from biotite.sequence.align.matrix import SubstitutionMatrix
    from biotite.sequence.sequence import Sequence

# The class is implemented in Rust and therefore is not generic at runtime;
# allow ``SeedExtension[S1, S2]`` subscription so the annotated stub works
SeedExtension.__class_getitem__ = classmethod(lambda cls, params: cls)


def align_local_ungapped(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix | tuple[int, int],
    seed: tuple[int, int],
    threshold: int,
    direction: str = "both",
    score_only: bool = False,
    check_matrix: bool = True,
) -> Alignment | int:
    """
    Perform a local alignment extending from given `seed` position
    without inserting gaps.

    .. deprecated::
        Use :class:`SeedExtension` instead.

    Parameters
    ----------
    seq1, seq2 : Sequence
        The sequences to be aligned.
    matrix : SubstitutionMatrix or tuple(int, int)
        Either a substitution matrix or a ``(match, mismatch)`` pair of scores.
    seed : tuple(int, int)
        The indices in `seq1` and `seq2` where the local alignment
        starts.
        The indices must be non-negative.
    threshold : int
        If the current score falls this value below the maximum score
        found, the alignment terminates.
    direction : {'both', 'upstream', 'downstream'}, optional
        Controls in which direction the alignment extends starting
        from the seed (see :class:`SeedExtension`).
    score_only : bool, optional
        If set to ``True``, only the similarity score is returned
        instead of the :class:`Alignment`.
    check_matrix : bool, optional
        If set to False, the `matrix` is not checked for compatibility
        with the alphabets of the sequences.

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
    SeedExtension
        The non-deprecated, reusable replacement for this function.
    align_local_gapped
        For gapped local alignments with the same *X-Drop* technique.
    """
    warnings.warn(
        "'align_local_ungapped()' is deprecated, use 'SeedExtension' instead",
        DeprecationWarning,
    )
    # Preserve the historical alphabet check of this function
    if check_matrix and not isinstance(matrix, tuple):
        if not matrix.get_alphabet1().extends(
            seq1.get_alphabet()
        ) or not matrix.get_alphabet2().extends(seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    extension = SeedExtension(matrix, threshold, direction)
    return extension.align(seq1, seq2, seed, score_only=score_only)
