# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Tests that apply equally to all three pairwise alignment functions
(:func:`align_optimal()`, :func:`align_banded()` and
:func:`align_local_gapped()`).

Both tests are driven by the legacy consistency data set, whose entries already
span all three functions and their parameter combinations. Tests specific to a
single function live in the respective ``test_<name>.py``.
"""

import numpy as np
import pytest
import biotite.sequence.align as align
from tests.sequence.align.util import legacy_alignments

_LEGACY_ALIGNMENTS = legacy_alignments()
_PARAMS = [param for param, _ in _LEGACY_ALIGNMENTS]
_IDS = [param["file_name"] for param in _PARAMS]


def _kwargs(param):
    """
    The alignment keyword arguments for a legacy `param` entry.

    Sequence-valued parameters (``gap_penalty``, ``band``, ``seed``) are stored
    as lists in JSON, but the functions expect tuples.
    """
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in param["params"].items()
    }


def _swap_order(kwargs):
    """
    Adapt the order-dependent parameters for swapped input sequences: the band
    diagonals (``diag = j - i``) flip sign and the seed coordinates swap.
    """
    swapped = dict(kwargs)
    if "band" in swapped:
        lower, upper = swapped["band"]
        swapped["band"] = (-upper, -lower)
    if "seed" in swapped:
        pos1, pos2 = swapped["seed"]
        swapped["seed"] = (pos2, pos1)
    return swapped


# Upper bound on the number of co-optimal alignments retrieved for set-based
# comparisons. If a function returns this many, the set may be truncated and the
# alignment-level comparison is skipped (only the score is compared).
MAX_NUMBER = 100


def _gapped_set(alignments):
    """The set of ``(gapped_seq1, gapped_seq2)`` tuples of the given alignments."""
    return {tuple(alignment.get_gapped_sequences()) for alignment in alignments}


def _center_seed(alignment):
    """The central aligned (non-gap) position of `alignment`, used as a seed."""
    trace = alignment.trace
    trace = trace[(trace != -1).all(axis=1)]
    return tuple(int(pos) for pos in trace[len(trace) // 2])


@pytest.mark.parametrize("param", _PARAMS, ids=_IDS)
def test_scoring(sequences, param):
    """
    The score reported in the returned alignment must be reproduced both by
    the standalone :func:`score()` function and by the ``score_only`` mode of
    the alignment function itself.
    """
    align_function = getattr(align, param["method"])
    i, j = param["seq_indices"]
    seq1, seq2 = sequences[i], sequences[j]
    kwargs = _kwargs(param)
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    alignment = align_function(seq1, seq2, matrix, max_number=1, **kwargs)[0]

    # The standalone score() function reproduces the alignment score
    # `terminal_penalty` only applies to align_optimal(); the other functions
    # produce no terminal gaps, so the default is irrelevant for them
    assert (
        align.score(
            alignment,
            matrix,
            kwargs["gap_penalty"],
            terminal_penalty=kwargs.get("terminal_penalty", False),
        )
        == alignment.score
    )
    # The score-only mode returns the same score as the full alignment
    assert (
        align_function(seq1, seq2, matrix, max_number=1, score_only=True, **kwargs)
        == alignment.score
    )


@pytest.mark.parametrize("param", _PARAMS, ids=_IDS)
def test_match_mismatch(sequences, param):
    """
    A ``(match, mismatch)`` scoring scheme must give the same result as a
    :class:`SubstitutionMatrix` encoding the same scheme: the match score on the
    diagonal and the mismatch score everywhere else.
    """
    MATCH = 5
    MISMATCH = -5

    align_function = getattr(align, param["method"])
    i, j = param["seq_indices"]
    seq1, seq2 = sequences[i], sequences[j]
    kwargs = _kwargs(param)

    # An equivalent substitution matrix over the (shared) sequence alphabet
    alphabet = seq1.get_alphabet()
    matrix_data = np.full((len(alphabet), len(alphabet)), MISMATCH, dtype=np.int32)
    np.fill_diagonal(matrix_data, MATCH)
    matrix = align.SubstitutionMatrix(alphabet, alphabet, matrix_data)
    from_matrix = align_function(seq1, seq2, matrix, max_number=1, **kwargs)[0]

    from_tuple = align_function(seq1, seq2, (MATCH, MISMATCH), max_number=1, **kwargs)[
        0
    ]

    assert from_tuple.score == from_matrix.score
    assert from_tuple.get_gapped_sequences() == from_matrix.get_gapped_sequences()


@pytest.mark.parametrize("param", _PARAMS, ids=_IDS)
def test_symmetry(sequences, param):
    """
    Swapping the two input sequences (and the order-dependent ``band``/``seed``
    parameters) must yield the same alignment, with the two aligned rows
    swapped.
    """
    align_function = getattr(align, param["method"])
    i, j = param["seq_indices"]
    seq1, seq2 = sequences[i], sequences[j]
    kwargs = _kwargs(param)
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    alignments = align_function(seq1, seq2, matrix, max_number=MAX_NUMBER, **kwargs)
    swapped = align_function(
        seq2, seq1, matrix, max_number=MAX_NUMBER, **_swap_order(kwargs)
    )

    assert alignments[0].score == swapped[0].score
    # The trace cannot be compared directly: with multiple co-optimal alignments
    # the two orderings may return different ones. Compare the full set of
    # alignments instead, but only if neither was truncated to `MAX_NUMBER`.
    if len(alignments) < MAX_NUMBER and len(swapped) < MAX_NUMBER:
        swapped_rows = {(row2, row1) for row1, row2 in _gapped_set(swapped)}
        assert _gapped_set(alignments) == swapped_rows


# align_optimal() is the reference here, so it cannot be compared to itself
_HEURISTIC_PARAMS = [param for param in _PARAMS if param["method"] != "align_optimal"]


@pytest.mark.parametrize(
    "param", _HEURISTIC_PARAMS, ids=[param["file_name"] for param in _HEURISTIC_PARAMS]
)
def test_comparison_to_optimal_alignment(sequences, param):
    """
    Given a sufficiently large search space, align_banded() and
    align_local_gapped() must recover the same alignment(s) as align_optimal().
    """
    i, j = param["seq_indices"]
    seq1, seq2 = sequences[i], sequences[j]
    kwargs = _kwargs(param)
    gap_penalty = kwargs["gap_penalty"]
    # `align_local_gapped()` is always local; `align_banded()` has `local` as parameter
    local = kwargs.get("local", True)
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    # Neither heuristic penalizes terminal gaps -> semi-global reference
    ref_alignments = align.align_optimal(
        seq1,
        seq2,
        matrix,
        gap_penalty=gap_penalty,
        local=local,
        terminal_penalty=False,
        max_number=MAX_NUMBER,
    )
    ref_alignments = [align.remove_terminal_gaps(ali) for ali in ref_alignments]

    if param["method"] == "align_banded":
        # A band wide enough to contain every diagonal
        test_alignments = align.align_banded(
            seq1,
            seq2,
            matrix,
            band=(-len(seq1), len(seq2)),
            gap_penalty=gap_penalty,
            local=local,
            max_number=MAX_NUMBER,
        )
    else:
        # A seed on the optimal local alignment and a threshold large enough to
        # not prune it
        test_alignments = align.align_local_gapped(
            seq1,
            seq2,
            matrix,
            seed=_center_seed(ref_alignments[0]),
            threshold=1000,
            gap_penalty=gap_penalty,
            max_number=MAX_NUMBER,
        )

    assert test_alignments[0].score == ref_alignments[0].score
    # Compare the alignments themselves, but only if neither set was truncated
    if len(test_alignments) < MAX_NUMBER and len(ref_alignments) < MAX_NUMBER:
        ref_gapped = _gapped_set(ref_alignments)
        for alignment in test_alignments:
            if param["method"] == "align_banded":
                assert tuple(alignment.get_gapped_sequences()) in ref_gapped
            else:
                if tuple(alignment.get_gapped_sequences()) not in ref_gapped:
                    # align_local_gapped() may legitimately differ:
                    # it can extend the upstream side slightly further than align_optimal()
                    # (an equal-scoring, longer alignment)
                    max_ref_length = max(len(ali) for ali in ref_alignments)
                    assert len(alignment) > max_ref_length


@pytest.mark.parametrize(
    "param, ref_alignment",
    _LEGACY_ALIGNMENTS,
    ids=_IDS,
)
def test_consistency_with_legacy(sequences, param, ref_alignment):
    """
    Verify that the current (Rust) implementation of each alignment function
    produces the same results as the legacy *Cython* implementation.
    """
    align_function = getattr(align, param["method"])
    i, j = param["seq_indices"]
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    test_alignment = align_function(
        sequences[i], sequences[j], matrix, max_number=1, **_kwargs(param)
    )[0]

    # Direct alignment comparison is not possible, because the reference
    # alignment loaded from FASTA has no score and the trace offset from
    # semi-global/local alignment is lost during FASTA round-tripping
    assert test_alignment.get_gapped_sequences() == ref_alignment.get_gapped_sequences()
