# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application_v2.viennarna import (
    CofoldedAlignment,
    CofoldedRNA,
    FoldedAlignment,
    FoldedRNA,
)
from biotite.application_v2.viennarna.result import _CANONICAL_PAIRS

# Long enough that the random generators reliably produce a rich structure
LENGTH = 40


def _pair_set(base_pairs):
    """Turn a base pair array into a set of tuples for order-independent comparison."""
    return {tuple(int(position) for position in row) for row in base_pairs}


def _random_nested_pairs(length, rng):
    """
    Generate random non-crossing base pairs over positions ``0..length-1``.
    """
    stack = []
    pairs = []
    for position in range(length):
        draw = rng.random()
        if stack and draw < 0.3:
            pairs.append((stack.pop(), position))
        elif draw < 0.65:
            stack.append(position)
    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def _random_matching(length, rng):
    """
    Generate a random base pair set that may contain crossing pairs.
    """
    P_PAIRS = 0.25

    positions = rng.permutation(length)
    n_pairs = int(length * P_PAIRS)
    pairs = positions[: 2 * n_pairs].reshape(-1, 2)
    return np.sort(pairs, axis=1)


def _random_dot_bracket(length, rng):
    """
    Generate a random nested (crossing-free) dot-bracket string.
    """
    symbols = np.full(length, ".")
    for opening, closing in _random_nested_pairs(length, rng):
        symbols[opening] = "("
        symbols[closing] = ")"
    return "".join(symbols)


def _random_multi_strand_dot_bracket(length, rng):
    """
    Generate a random nested dot-bracket string split into two strands.
    """
    dot_bracket = _random_dot_bracket(length, rng)
    split = length // 2
    return dot_bracket[:split] + "&" + dot_bracket[split:]


def _canonical_symbols(length, pairs):
    """
    Assign symbols so that each base pair is a canonical ``G-C`` pair.
    """
    symbols = np.full(length, "A")
    for opening, closing in pairs:
        symbols[opening] = "G"
        symbols[closing] = "C"
    return symbols


def _no_crossing(base_pairs):
    """
    Check that no two base pairs cross each other.
    """
    pairs = base_pairs.tolist()
    for a, b in pairs:
        for c, d in pairs:
            if a < c < b < d:
                return False
    return True


def _ungapped_alignment(symbols):
    """
    Create a two-sequence alignment without gaps from a symbol array.
    """
    sequence = seq.NucleotideSequence("".join(symbols))
    trace = np.column_stack([np.arange(len(symbols)), np.arange(len(symbols))])
    return align.Alignment([sequence, sequence], trace)


def _folded_from_base_pairs(cls, pairs, symbols, **kwargs):
    """
    Build a :class:`FoldedRNA` or :class:`FoldedAlignment` via its
    ``from_base_pairs()`` constructor from a symbol array.
    """
    if cls is FoldedRNA:
        container = seq.NucleotideSequence("".join(symbols))
    else:
        container = _ungapped_alignment(symbols)
    return cls.from_base_pairs(container, pairs, **kwargs)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("cls", [FoldedRNA, FoldedAlignment])
def test_from_base_pairs_round_trip(cls, seed):
    """
    ``from_base_pairs()`` is the inverse of ``base_pairs()``:
    the base pairs a structure is built from are recovered unchanged.
    """
    rng = np.random.default_rng(seed)
    pairs = _random_nested_pairs(LENGTH, rng)
    symbols = _canonical_symbols(LENGTH, pairs)
    folded = _folded_from_base_pairs(cls, pairs, symbols)

    assert _pair_set(folded.base_pairs()) == _pair_set(pairs)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_pseudoknot_order", [0, None])
@pytest.mark.parametrize("cls", [FoldedRNA, FoldedAlignment])
def test_from_base_pairs_postconditions(cls, max_pseudoknot_order, seed):
    """
    ``from_base_pairs()`` keeps only base pairs that are canonical for the
    given sequence, and ``max_pseudoknot_order=0`` additionally removes all
    crossing base pairs.
    """
    rng = np.random.default_rng(seed)
    pairs = _random_matching(LENGTH, rng)
    symbols = rng.choice(["A", "C", "G", "T"], size=LENGTH)
    folded = _folded_from_base_pairs(
        cls, pairs, symbols, max_pseudoknot_order=max_pseudoknot_order
    )

    recovered = folded.base_pairs()
    # The random symbols retain a non-trivial number of canonical pairs
    assert len(recovered) > 0
    for opening, closing in recovered:
        assert symbols[opening] + symbols[closing] in _CANONICAL_PAIRS

    if max_pseudoknot_order == 0:
        assert _no_crossing(folded.base_pairs())


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("cls", [FoldedRNA, FoldedAlignment])
def test_span_is_translation(cls, seed):
    """
    Setting a ``span`` merely translates the base pair positions by the span offset.
    """
    rng = np.random.default_rng(seed)
    dot_bracket = _random_dot_bracket(LENGTH, rng)
    span = tuple(sorted(rng.integers(LENGTH, size=2)))
    if cls is FoldedRNA:
        container = seq.NucleotideSequence("A" * LENGTH)
    else:
        container = _ungapped_alignment(np.full(LENGTH, "A"))
    without_span = cls(container, dot_bracket)
    with_span = cls(container, dot_bracket, span=span)

    assert np.array_equal(with_span.base_pairs(), without_span.base_pairs() + span[0])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("cls", [CofoldedRNA, CofoldedAlignment])
def test_spans_is_translation(cls, seed):
    """
    Setting per-strand ``spans`` translates each strand's positions by its own offset,
    leaving the strand assignment untouched.
    """
    rng = np.random.default_rng(seed)
    dot_bracket = _random_multi_strand_dot_bracket(LENGTH, rng)
    segments = dot_bracket.split("&")
    spans = [tuple(sorted(rng.integers(LENGTH, size=2))) for _ in segments]
    offsets = np.array([span[0] for span in spans])
    if cls is CofoldedRNA:
        strands = [seq.NucleotideSequence("A" * len(segment)) for segment in segments]
    else:
        strands = [_ungapped_alignment(np.full(len(s), "A")) for s in segments]
    without_spans = cls(strands, dot_bracket)
    with_spans = cls(strands, dot_bracket, spans=spans)

    without_span_pairs = without_spans.base_pairs()
    expected = without_span_pairs.copy()
    expected[:, 1] += offsets[without_span_pairs[:, 0]]
    expected[:, 3] += offsets[without_span_pairs[:, 2]]
    assert np.array_equal(with_spans.base_pairs(), expected)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("cls", [CofoldedRNA, CofoldedAlignment])
def test_from_single_embedding(cls, seed):
    """
    ``from_single()`` embeds a single-strand result as a one-strand complex, preserving
    both the base pairs and the metadata.
    """
    rng = np.random.default_rng(seed)
    dot_bracket = _random_dot_bracket(LENGTH, rng)
    if cls is CofoldedRNA:
        single = FoldedRNA(
            seq.NucleotideSequence("A" * LENGTH), dot_bracket, free_energy=-1.5
        )
        cofolded = cls.from_single(single)
        metadata_ok = (
            cofolded.sequences == [single.sequence]
            and cofolded.free_energy == single.free_energy
            and cofolded.spans is None
        )
    else:
        single = FoldedAlignment(
            _ungapped_alignment(np.full(LENGTH, "A")),
            dot_bracket,
            free_energy=-1.5,
            covariance_energy=-0.5,
        )
        cofolded = cls.from_single(single)
        metadata_ok = (
            cofolded.alignments == [single.alignment]
            and cofolded.free_energy == single.free_energy
            and cofolded.covariance_energy == single.covariance_energy
            and cofolded.spans is None
        )
    single_pairs = single.base_pairs()
    cofolded_pairs = cofolded.base_pairs()

    # The complex contains only one strand -> both strand channels are 0
    assert np.all(cofolded_pairs[:, 0] == 0)
    assert np.all(cofolded_pairs[:, 2] == 0)
    # The position channels reproduce the single-strand base pairs
    assert _pair_set(cofolded_pairs[:, [1, 3]]) == _pair_set(single_pairs)
    assert metadata_ok


@pytest.mark.parametrize(
    ["folded", "n_channels"],
    [
        pytest.param(
            FoldedRNA(seq.NucleotideSequence("AAAA"), "...."), 2, id="FoldedRNA"
        ),
        pytest.param(
            CofoldedRNA(
                [seq.NucleotideSequence("AA"), seq.NucleotideSequence("AA")], "..&.."
            ),
            4,
            id="CofoldedRNA",
        ),
        pytest.param(
            FoldedAlignment(_ungapped_alignment(np.full(4, "A")), "...."),
            2,
            id="FoldedAlignment",
        ),
        pytest.param(
            CofoldedAlignment(
                [_ungapped_alignment(np.full(2, "A")) for _ in range(2)], "..&.."
            ),
            4,
            id="CofoldedAlignment",
        ),
    ],
)
def test_empty_base_pairs(folded, n_channels):
    """
    A structure without base pairs yields a correctly shaped and typed
    empty array.
    """
    base_pairs = folded.base_pairs()
    assert base_pairs.shape == (0, n_channels)
    assert np.issubdtype(base_pairs.dtype, np.integer)


def test_cofolded_rna_channel_layout():
    """
    The 4-channel base pair layout is ``(strand_i, pos_i, strand_j, pos_j)``,
    here for an inter-strand pairing between two strands.
    """
    sequences = [seq.NucleotideSequence("GG"), seq.NucleotideSequence("CC")]
    cofolded = CofoldedRNA(sequences, "((&))")
    assert cofolded.base_pairs().tolist() == [[0, 0, 1, 1], [0, 1, 1, 0]]
