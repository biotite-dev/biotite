# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.sequence.align"
__author__ = "Patrick Kunzmann"
__all__ = ["align_local_gapped"]

import itertools
from typing import TYPE_CHECKING, Literal, overload
import numpy as np
from biotite.rust.sequence.align import align_region as _rust_align_region
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.align.pairwise import prepare_scoring

if TYPE_CHECKING:
    from biotite.sequence.align.matrix import SubstitutionMatrix
    from biotite.sequence.sequence import Sequence
    from biotite.typing import S1, S2


@overload
def align_local_gapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    seed: tuple[int, int],
    threshold: int,
    gap_penalty: int | tuple[int, int] = -10,
    max_number: int = 1,
    direction: Literal["both", "upstream", "downstream"] = "both",
    score_only: Literal[False] = False,
    max_table_size: int | None = None,
) -> list[Alignment]: ...
@overload
def align_local_gapped(
    seq1: Sequence[S1],
    seq2: Sequence[S2],
    matrix: SubstitutionMatrix[S1, S2] | tuple[int, int],
    seed: tuple[int, int],
    threshold: int,
    gap_penalty: int | tuple[int, int] = -10,
    max_number: int = 1,
    direction: Literal["both", "upstream", "downstream"] = "both",
    *,
    score_only: Literal[True],
    max_table_size: int | None = None,
) -> int: ...
def align_local_gapped(
    seq1: Sequence,
    seq2: Sequence,
    matrix: SubstitutionMatrix | tuple[int, int],
    seed: tuple[int, int],
    threshold: int,
    gap_penalty: int | tuple[int, int] = -10,
    max_number: int = 1,
    direction: str = "both",
    score_only: bool = False,
    max_table_size: int | None = None,
) -> list[Alignment] | int:
    """
    Perform a local gapped alignment extending from a given `seed`
    position.

    The alignment extends into one or both directions (controlled by
    `direction`) until the total alignment score falls more than
    `threshold` below the maximum score found (*X-Drop*).
    :footcite:`Zhang2000`
    The returned alignment contains the range that yielded the maximum
    score.

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
    gap_penalty : int or tuple(int, int), optional
        If an integer is provided, the value will be interpreted as
        linear gap penalty.
        If a tuple is provided, an affine gap penalty is used
        :footcite:`Gotoh1982`.
        The first integer in the tuple is the gap opening penalty,
        the second integer is the gap extension penalty.
        The values need to be negative.
    max_number : int, optional
        The maximum number of alignments returned.
        When the number of branches exceeds this value in the traceback
        step, no further branches are created.
        By default, only a single alignment is returned.
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
    max_table_size : int, optional
        A :class:`MemoryError` is raised, if the number of cells
        in the internal dynamic programming table, i.e. approximately
        the product of the lengths of the aligned regions, would exceed
        the given value.

    Returns
    -------
    alignments : list of Alignment
        A list of found alignments.
        Each alignment in the list has the same similarity
        score.
        Only returned, if `score_only` is ``False``.
    score : int
        The alignment similarity score.
        Only returned, if `score_only` is ``True``.

    See Also
    --------
    align_optimal
        For the optimal (non-heuristic) alignment of two sequences.
    align_banded
        For a heuristic alignment constrained to a diagonal band.

    Notes
    -----
    Unlike :func:`align_optimal()`, this function does not allocate
    memory proportional to the length of both sequences, but only
    approximately proportional to lengths of the aligned regions.
    In principle, this makes this function viable for local alignments
    of sequences of any length.
    However, if the product of the lengths of the homologous regions
    is too large to fit into memory, a :class:`MemoryError` or even a
    crash may occur.
    This may also happen in spurious long alignments due to poor choice
    of substitution matrix or gap penalty.
    You may set `max_table_size` to avoid excessive memory use and
    crashes.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> seq1 = NucleotideSequence("CGTAGCTATCGCCTGTACGGTT")
    >>> seq2 = NucleotideSequence("TATATGCCTTACGGAATTGCTTTTT")
    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> alignment = align_local_gapped(
    ...     seq1, seq2, matrix, seed=(16, 10), threshold=20
    ... )[0]
    >>> print(alignment)
    TATCGCCTGTACGG
    TAT-GCCT-TACGG
    """
    # Resolve the scoring scheme (substitution matrix or match/mismatch tuple)
    scoring = prepare_scoring(matrix, seq1, seq2)

    # Check if gap penalty is linear or affine
    if isinstance(gap_penalty, int):
        if gap_penalty >= 0:
            raise ValueError("Gap penalty must be negative")
    elif isinstance(gap_penalty, tuple):
        if gap_penalty[0] >= 0 or gap_penalty[1] >= 0:
            raise ValueError("Gap penalty must be negative")
    else:
        raise TypeError("Gap penalty must be either integer or tuple")

    # Check if max_number is reasonable
    if max_number < 1:
        raise ValueError("Maximum number of returned alignments must be at least 1")

    # Check maximum table size
    if max_table_size is None:
        max_table_size = np.iinfo(np.int64).max
    elif max_table_size <= 0:
        raise ValueError("Maximum table size must be a positve value")

    seq1_start, seq2_start = seed
    if seq1_start < 0 or seq2_start < 0:
        raise IndexError("Seed must contain positive indices")
    if seq1_start >= len(seq1) or seq2_start >= len(seq2):
        raise IndexError(
            f"Seed {(seq1_start, seq2_start)} is out of bounds "
            f"for the sequences of length {len(seq1)} and {len(seq2)}"
        )

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
    # Range check to avoid negative indices
    if seq1_start == 0 or seq2_start == 0:
        upstream = False

    if threshold < 0:
        raise ValueError("The threshold value must be a non-negative integer")

    # The two sequences may use different code dtypes
    # -> widen both to a common type so the Rust layer dispatches once
    common_dtype = np.promote_types(seq1.code.dtype, seq2.code.dtype)
    code1 = np.ascontiguousarray(seq1.code, common_dtype)
    code2 = np.ascontiguousarray(seq2.code, common_dtype)

    total_score = 0
    upstream_traces = []
    downstream_traces = []
    # Separate alignment into two parts:
    # the regions upstream and downstream from the seed position
    if upstream:
        # For the upstream region the respective part of the sequence
        # must be reversed
        traces, score = _rust_align_region(
            np.ascontiguousarray(code1[seq1_start - 1 :: -1]),
            np.ascontiguousarray(code2[seq2_start - 1 :: -1]),
            scoring,
            gap_penalty,
            threshold,
            max_number,
            score_only,
            int(max_table_size),
        )
        total_score += score
        if not score_only:
            # Undo the sequence reversing
            upstream_traces = [trace[::-1] for trace in traces]
            offset = np.array(seed) - 1
            for trace in upstream_traces:
                # Gap values (-1) are not transformed,
                # as gaps are not indices
                non_gap_mask = trace != -1
                # Second part of sequence reversing
                trace[non_gap_mask] *= -1
                # Add seed offset to trace indices
                trace[non_gap_mask[:, 0], 0] += offset[0]
                trace[non_gap_mask[:, 1], 1] += offset[1]

    if downstream:
        traces, score = _rust_align_region(
            np.ascontiguousarray(code1[seq1_start + 1 :]),
            np.ascontiguousarray(code2[seq2_start + 1 :]),
            scoring,
            gap_penalty,
            threshold,
            max_number,
            score_only,
            int(max_table_size),
        )
        total_score += score
        if not score_only:
            downstream_traces = traces
            offset = np.array(seed) + 1
            for trace in downstream_traces:
                trace[trace[:, 0] != -1, 0] += offset[0]
                trace[trace[:, 1] != -1, 1] += offset[1]

    # Add the score of the seed position itself
    if isinstance(scoring, tuple):
        match_score, mismatch_score = scoring
        total_score += (
            match_score if code1[seq1_start] == code2[seq2_start] else mismatch_score
        )
    else:
        total_score += int(scoring[code1[seq1_start], code2[seq2_start]])

    if score_only:
        return total_score

    if upstream and downstream:
        # Create cartesian product of upstream and downstream traces
        # Only consider max_number alignments
        traces = [
            np.concatenate([upstream_trace, [seed], downstream_trace])
            for _, (upstream_trace, downstream_trace) in zip(
                range(max_number),
                itertools.product(upstream_traces, downstream_traces),
            )
        ]
    elif upstream:
        traces = [np.concatenate([trace, [seed]]) for trace in upstream_traces]
    elif downstream:
        traces = [np.concatenate([[seed], trace]) for trace in downstream_traces]
    else:
        # 'direction == "upstream"', but the start index is 0 so no
        # upstream alignment is performed
        # -> the trace includes only the seed
        traces = [np.array(seed)[np.newaxis, :]]

    return [Alignment([seq1, seq2], trace, total_score) for trace in traces]
