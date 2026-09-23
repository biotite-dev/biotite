import numpy as np
import pytest
import biotite.application_v2.muscle as muscle
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application_v2 import VersionError
from tests.util import is_not_installed


@pytest.mark.skipif(is_not_installed("muscle"), reason="MUSCLE is not installed")
@pytest.mark.parametrize("gap_penalty", [-10, (-10, -1)])
def test_align_multiple(sequences, gap_penalty):
    r"""
    Test `align_multiple()` function using actual long sequences,
    compared to the output of MUSCLE.
    Both alignment methods are heuristic, the exact same result is not
    expected.
    Just assert that the resulting score is at least the 50 % of the
    score of the MUSCLE alignment.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    test_alignment, order, tree, distances = align.align_multiple(
        sequences, matrix, gap_penalty=gap_penalty, terminal_penalty=True
    )
    test_score = align.score(test_alignment, matrix, gap_penalty, terminal_penalty=True)

    try:
        ref_alignment = (
            muscle.Muscle3App()
            .run(sequences, matrix=matrix, gap_penalty=gap_penalty)
            .result()
            .alignment
        )
    except VersionError:
        pytest.skip("Invalid Muscle software version")
    ref_score = align.score(ref_alignment, matrix, gap_penalty, terminal_penalty=True)

    assert test_score >= ref_score * 0.5


@pytest.mark.parametrize("scores", [(1, -1), (5, -4)])
def test_match_mismatch_scores(scores):
    """
    A ``(match, mismatch)`` tuple gives the same result as the
    equivalent substitution matrix.
    """
    rng = np.random.default_rng(0)
    alphabet = seq.NucleotideSequence.alphabet_unamb
    # Related sequences: random point mutations of a common ancestor
    ancestor = rng.choice(alphabet.get_symbols(), size=50)
    sequences = []
    for _ in range(5):
        mutant = ancestor.copy()
        positions = rng.choice(len(ancestor), size=5, replace=False)
        mutant[positions] = rng.choice(alphabet.get_symbols(), size=5)
        sequences.append(seq.NucleotideSequence("".join(mutant)))
    match, mismatch = scores
    score_matrix = np.full((len(alphabet), len(alphabet)), mismatch)
    np.fill_diagonal(score_matrix, match)
    matrix = align.SubstitutionMatrix(alphabet, alphabet, score_matrix)

    test_alignment, _, _, _ = align.align_multiple(sequences, scores)
    ref_alignment, _, _, _ = align.align_multiple(sequences, matrix)

    assert test_alignment == ref_alignment
