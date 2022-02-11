# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import biotite.sequence.align as align
import biotite.application.muscle as muscle
from biotite.application import VersionError
from ...util import is_not_installed
from .util import sequences



@pytest.mark.skipif(
    is_not_installed("muscle"),
    reason="MUSCLE is not installed"
)
@pytest.mark.parametrize("gap_penalty", [-10, (-10,-1)])
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
    test_score = align.score(
        test_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    
    try:
        ref_alignment = muscle.MuscleApp.align(
            sequences, matrix=matrix, gap_penalty=gap_penalty
        )
    except VersionError:
        pytest.skip(f"Invalid Muscle software version")
    ref_score = align.score(
        ref_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    
    assert test_score >= ref_score * 0.5