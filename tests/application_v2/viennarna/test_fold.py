# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import dataclasses
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application_v2.viennarna import (
    AliduplexApp,
    AlifoldApp,
    CofoldApp,
    CofoldedAlignment,
    CofoldedRNA,
    DuplexApp,
    EvalApp,
    FoldApp,
    FoldedAlignment,
    FoldedRNA,
    LalifoldApp,
    LfoldApp,
    MultifoldApp,
    PKplexApp,
    PlexApp,
    SuboptApp,
)
from biotite.structure.dotbracket import base_pairs_from_dot_bracket
from tests.util import is_not_installed

# Each implemented folding `LocalApp` together with the name of its executable
BIN_PATH = {
    FoldApp: "RNAfold",
    SuboptApp: "RNAsubopt",
    PKplexApp: "RNAPKplex",
    LfoldApp: "RNALfold",
    CofoldApp: "RNAcofold",
    MultifoldApp: "RNAmultifold",
    DuplexApp: "RNAduplex",
    PlexApp: "RNAplex",
    AlifoldApp: "RNAalifold",
    LalifoldApp: "RNALalifold",
    AliduplexApp: "RNAaliduplex",
    EvalApp: "RNAeval",
}


def _app_or_skip(app_class):
    """
    Return a folding app instance or skip the test if it is not available.
    """
    bin_path = BIN_PATH[app_class]
    if is_not_installed(bin_path):
        pytest.skip(f"'{bin_path}' is not installed")
    return app_class(bin_path)


def _random_sequence(rng, length):
    """
    A random nucleotide sequence.
    """
    return seq.NucleotideSequence(rng.choice(["A", "C", "G", "T"], size=length))


def _random_hairpin(rng, stem=7, loop=5):
    """
    A sequence that reliably folds: stem · loop · reverse-complement(stem).
    """
    stem_seq = _random_sequence(rng, stem)
    return stem_seq + _random_sequence(rng, loop) + stem_seq.reverse().complement()


def _self_alignment(sequence):
    """
    An alignment of a sequence with itself (no covariation).
    """
    return align.align_ungapped(
        sequence, sequence, align.SubstitutionMatrix.std_nucleotide_matrix()
    )


def _gapped_self_alignment(sequence, rng):
    """
    An alignment of a sequence with itself with an all-gap column inserted.
    As the gap occupies the same column in both rows, the consensus structure
    is unchanged, but projecting it onto the sequence must now remap alignment
    columns to ungapped residue positions instead of being the identity.
    """
    length = len(sequence)
    gap_column = int(rng.integers(1, length))
    trace = np.column_stack([np.arange(length), np.arange(length)])
    trace = np.insert(trace, gap_column, -1, axis=0)
    return align.Alignment([sequence, sequence], trace)


def _random_hairpin_pairs(length, rng):
    """
    A random nested stem of base pairs, guaranteed non-empty with a loop >= 3.
    """
    stem = int(rng.integers(2, 6))
    loop = int(rng.integers(3, 6))
    start = int(rng.integers(0, length - 2 * stem - loop + 1))
    close = start + 2 * stem + loop - 1
    return np.array([[start + i, close - i] for i in range(stem)])


def _pair_set(base_pairs):
    """
    Turn a base pair array into a set of tuples for order-independent comparison.
    """
    return {tuple(int(position) for position in row) for row in base_pairs}


def _as_list(results):
    """
    Wrap a single result into a list, leaving a list of results untouched.
    Required as SuboptApp returns a `list[FoldedRNA]`.
    """
    return results if isinstance(results, list) else [results]


def _assert_consistent_base_pairs(folded):
    """
    The base pairs of a single-strand result must be recoverable from
    its dot-bracket notation.
    """
    ref_base_pairs = base_pairs_from_dot_bracket(folded.dot_bracket)
    if folded.span is not None:
        ref_base_pairs = ref_base_pairs + folded.span[0]
    assert np.array_equal(folded.base_pairs(), ref_base_pairs)


def _assert_span_valid(result, full_length):
    """
    A ``span``, if present, must lie within the sequence and match the structure.
    """
    if result.span is None:
        assert len(result.dot_bracket) == full_length
    else:
        start, stop = result.span
        assert 0 <= start < stop <= full_length
        assert stop - start == len(result.dot_bracket)


def _assert_multi_strand(result, n_strands):
    """
    A multi-strand result must have valid four-column base pairs.
    """
    base_pairs = result.base_pairs()
    assert base_pairs.shape[1] == 4
    assert np.all((base_pairs[:, 0] >= 0) & (base_pairs[:, 0] < n_strands))
    assert np.all((base_pairs[:, 2] >= 0) & (base_pairs[:, 2] < n_strands))
    # The number of pairs is independent of the 4-channel derivation
    n_expected = len(base_pairs_from_dot_bracket(result.dot_bracket.replace("&", "")))
    assert len(base_pairs) == n_expected
    if result.spans is not None:
        assert len(result.spans) == n_strands


def _assert_well_formed(result):
    """
    A result must be internally consistent for its type.
    """
    match result:
        case FoldedRNA():
            _assert_consistent_base_pairs(result)
            _assert_span_valid(result, len(result.sequence))
            assert result.free_energy is None or isinstance(result.free_energy, float)
        case FoldedAlignment():
            _assert_consistent_base_pairs(result)
            _assert_span_valid(result, len(result.alignment))
            assert result.free_energy is None or isinstance(result.free_energy, float)
            assert isinstance(result.covariance_energy, float)
        case CofoldedRNA():
            _assert_multi_strand(result, len(result.sequences))
        case CofoldedAlignment():
            _assert_multi_strand(result, len(result.alignments))


def _assert_input_preserved(result, input):
    """
    The result must echo back the sequence(s)/alignment(s) it was run on.
    """
    match result:
        case FoldedRNA():
            assert result.sequence == input
        case CofoldedRNA():
            assert list(result.sequences) == list(input)
        case FoldedAlignment():
            assert result.alignment is input
        case CofoldedAlignment():
            assert result.alignments == input


### Input builders:
# Each builder returns the positional args, keyword args and the input that the
# result is expected to echo back, so a single test can cover every app.


def _build_fold(rng):
    sequence = _random_hairpin(rng)
    return (sequence,), {}, sequence


def _build_subopt(rng):
    sequence = _random_hairpin(rng)
    return (sequence,), {"energy_range": 2.0}, sequence


def _build_pkplex(rng):
    sequence = _random_hairpin(rng)
    return (sequence,), {}, sequence


def _build_lfold(rng):
    sequence = _random_sequence(rng, 60)
    return (sequence,), {"max_span": 40}, sequence


def _build_cofold(rng):
    sequence1, sequence2 = _random_hairpin(rng), _random_hairpin(rng)
    return (sequence1, sequence2), {}, [sequence1, sequence2]


def _build_multifold(rng):
    sequences = [_random_hairpin(rng) for _ in range(3)]
    return (sequences,), {}, sequences


def _build_duplex(rng):
    strand = _random_sequence(rng, 15)
    complement = strand.reverse().complement()
    return (strand, complement), {}, [strand, complement]


def _build_plex(rng):
    target, query = _random_sequence(rng, 30), _random_sequence(rng, 12)
    return (target, query), {}, [target, query]


def _build_alifold(rng):
    alignment = _self_alignment(_random_hairpin(rng))
    return (alignment,), {}, alignment


def _build_lalifold(rng):
    alignment = _self_alignment(_random_sequence(rng, 60))
    return (alignment,), {"max_span": 40}, alignment


def _build_aliduplex(rng):
    strand = _random_sequence(rng, 15)
    alignment1, alignment2 = (
        _self_alignment(strand),
        _self_alignment(strand.reverse().complement()),
    )
    return (alignment1, alignment2), {}, [alignment1, alignment2]


def _build_eval(rng):
    stem, loop = 7, 5
    sequence = _random_hairpin(rng, stem, loop)
    dot_bracket = "(" * stem + "." * loop + ")" * stem
    return (FoldedRNA(sequence, dot_bracket),), {}, sequence


INPUT_BUILDERS = {
    FoldApp: _build_fold,
    SuboptApp: _build_subopt,
    PKplexApp: _build_pkplex,
    LfoldApp: _build_lfold,
    CofoldApp: _build_cofold,
    MultifoldApp: _build_multifold,
    DuplexApp: _build_duplex,
    PlexApp: _build_plex,
    AlifoldApp: _build_alifold,
    LalifoldApp: _build_lalifold,
    AliduplexApp: _build_aliduplex,
    EvalApp: _build_eval,
}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("app_class", list(INPUT_BUILDERS))
def test_application_success(app_class, seed):
    """
    Every app runs on a random input and returns a result that is consistent with its
    input.
    """
    app = _app_or_skip(app_class)
    args, kwargs, input = INPUT_BUILDERS[app_class](np.random.default_rng(seed))

    for result in _as_list(app.run(*args, **kwargs).result()):
        _assert_input_preserved(result, input)
        _assert_well_formed(result)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "app_class", [FoldApp, SuboptApp, CofoldApp, LfoldApp, MultifoldApp]
)
def test_eval_reproduces_energy(app_class, seed):
    """
    Evaluating a structure with :class:`EvalApp` reproduces the free energy that
    the producing app assigned to it -- both share the Turner energy model.
    """
    producer, eval_app = _app_or_skip(app_class), _app_or_skip(EvalApp)
    args, kwargs, _ = INPUT_BUILDERS[app_class](np.random.default_rng(seed))

    for result in _as_list(producer.run(*args, **kwargs).result()):
        # Strip the energy to prove EvalApp computes it, not passes it through
        evaluated = eval_app.run(dataclasses.replace(result, free_energy=None)).result()
        assert evaluated.free_energy == result.free_energy


@pytest.mark.parametrize("seed", range(5))
def test_subopt_brackets_mfe(seed):
    """
    :class:`SuboptApp` returns the MFE structure among its suboptimal results,
    and every result lies within the requested energy range.
    """
    fold_app, subopt_app = _app_or_skip(FoldApp), _app_or_skip(SuboptApp)
    sequence = _random_hairpin(np.random.default_rng(seed))
    energy_range = 4.0

    mfe = fold_app.run(sequence).result()
    results = subopt_app.run(sequence, energy_range=energy_range).result()

    energies = [result.free_energy for result in results]
    assert min(energies) == mfe.free_energy
    # A small tolerance absorbs the floating-point error at the range boundary
    assert max(energies) <= mfe.free_energy + energy_range + 1e-6
    # The MFE structure is present among the suboptimal ones (may be degenerate)
    assert any(result.dot_bracket == mfe.dot_bracket for result in results)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ["ali_app_class", "single_app_class"],
    [
        pytest.param(AlifoldApp, FoldApp, id="alifold-fold"),
    ],
)
def test_alignment_projection_matches_single(ali_app_class, single_app_class, seed):
    """
    Projecting an alignment's consensus onto one sequence gives the same base
    pairs as the single-sequence app for that sequence: with identical rows the
    consensus collapses to the single-sequence structure, and a gap column in
    the alignment must be remapped away by the projection.
    """
    ali_app, single_app = _app_or_skip(ali_app_class), _app_or_skip(single_app_class)
    rng = np.random.default_rng(seed)
    strands = [_random_hairpin(rng)]
    # A gap column makes the projection remap columns to residue positions
    alignments = [_gapped_self_alignment(strands[0], rng)]

    single = single_app.run(*strands).result()
    consensus = ali_app.run(*alignments).result()

    assert _pair_set(consensus.base_pairs_of(0)) == _pair_set(single.base_pairs())


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("app_class", [CofoldApp, DuplexApp])
def test_perfect_complement_is_registered_duplex(app_class, seed):
    """
    A strand and its reverse complement hybridize into an in-register duplex:
    all base pairs are inter-molecular and pair position ``p`` with ``len-1-p``.
    """
    app = _app_or_skip(app_class)
    strand = _random_sequence(np.random.default_rng(seed), 18)

    base_pairs = app.run(strand, strand.reverse().complement()).result().base_pairs()
    assert len(base_pairs) > 0
    # All base pairs are inter-molecular
    assert np.all(base_pairs[:, 0] != base_pairs[:, 2])
    # Each base is paired with its reverse complement
    assert np.all(base_pairs[:, 1] + base_pairs[:, 3] == len(strand) - 1)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("app_class", [DuplexApp, PlexApp, AliduplexApp])
def test_only_intermolecular_pairs(app_class, seed):
    """
    The duplex-style apps form only inter-molecular base pairs.
    """
    app = _app_or_skip(app_class)
    args, kwargs, _ = INPUT_BUILDERS[app_class](np.random.default_rng(seed))

    for result in _as_list(app.run(*args, **kwargs).result()):
        match result:
            case CofoldedAlignment():
                base_pairs = result.base_pairs_of(0)
            case CofoldedRNA():
                base_pairs = result.base_pairs()
        assert np.all(base_pairs[:, 0] != base_pairs[:, 2])


@pytest.mark.parametrize("seed", range(5))
def test_fold_enforces_constraints(seed):
    """
    Enforcing base pairs on an all-``A`` sequence -- which cannot pair on its
    own -- reproduces exactly the constrained pairs and nothing else.
    """
    app = _app_or_skip(FoldApp)
    pairs = _random_hairpin_pairs(20, np.random.default_rng(seed))

    result = app.run(
        seq.NucleotideSequence("A" * 20), pairs=pairs, enforce=True
    ).result()
    assert _pair_set(result.base_pairs()) == _pair_set(pairs)


def test_subopt_respects_unpaired():
    """
    Positions constrained to be unpaired are ``.`` in every suboptimal
    structure.
    """
    app = _app_or_skip(SuboptApp)
    # A fixed short sequence is used deliberately: ``RNAsubopt`` aborts on many
    # constrained inputs, so a random sequence cannot be folded reliably
    sequence = seq.NucleotideSequence("GGGGAAAACCCC")
    unpaired = np.array([0, len(sequence) - 1])

    results = app.run(sequence, energy_range=3.0, unpaired=unpaired).result()
    assert len(results) > 0
    for result in results:
        assert result.dot_bracket[0] == "."
        assert result.dot_bracket[-1] == "."


@pytest.mark.parametrize("seed", range(5))
def test_eval_span_padding_is_energy_neutral(seed):
    """
    :class:`EvalApp` pads a local structure (with a ``span``) to the full sequence
    length, giving the same energy as the equivalent full-length structure.
    """
    MAX_STEM = 20
    MAX_LOOP = 20
    MAX_FLANK = 20

    eval_app = _app_or_skip(EvalApp)
    rng = np.random.default_rng(seed)
    stem = int(rng.integers(2, MAX_STEM + 1))
    loop = int(rng.integers(3, MAX_LOOP + 1))
    flank = int(rng.integers(1, MAX_FLANK + 1))
    hairpin = _random_hairpin(rng, stem, loop)
    sequence = _random_sequence(rng, flank) + hairpin + _random_sequence(rng, flank)
    local_structure = "(" * stem + "." * loop + ")" * stem

    span = (flank, flank + len(hairpin))
    local = eval_app.run(FoldedRNA(sequence, local_structure, span=span)).result()
    full = eval_app.run(
        FoldedRNA(sequence, "." * flank + local_structure + "." * flank)
    ).result()

    assert local.free_energy == full.free_energy
