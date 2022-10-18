# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application.viennarna import RNAalifoldApp
from ..util import is_not_installed


@pytest.fixture
def sample_app():
    """
    Provide a `RNAalifold` object, where *RNAalifold* has been executed for
    a sample alignment.
    """
    sequence = seq.NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    alignment = align.align_ungapped(sequence, sequence, matrix)
    app = RNAalifoldApp(alignment)
    app.start()
    app.join()
    return app


@pytest.mark.skipif(
    is_not_installed("RNAalifold"), reason="RNAalifold is not installed"
)
def test_get_dot_bracket(sample_app):
    assert sample_app.get_dot_bracket() ==  "(((.((((.......)).)))))...."


@pytest.mark.skipif(
    is_not_installed("RNAalifold"), reason="RNAalifold is not installed"
)
def test_get_free_energy(sample_app):
    assert sample_app.get_free_energy() == -1.3

@pytest.mark.skipif(
    is_not_installed("RNAalifold"), reason="RNAalifold is not installed"
)
def test_get_base_pairs(sample_app):
    expected_basepairs = np.array([[ 0, 22],
                                   [ 1, 21],
                                   [ 2, 20],
                                   [ 4, 19],
                                   [ 5, 18],
                                   [ 6, 16],
                                   [ 7, 15]])
    assert np.all(sample_app.get_base_pairs() == expected_basepairs)

@pytest.mark.skipif(
    is_not_installed("RNAalifold"), reason="RNAalifold is not installed"
)
def test_constraints():
    """
    Constrain every position of the input alignment and expect that the
    same base pairs are returned independent of the sequence.
    """
    # Sequence should not matter
    sequence = seq.NucleotideSequence("A" * 20)
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    alignment = align.align_ungapped(sequence, sequence, matrix)
    
    # An arbitrary secondary structure
    # The loop in the center must probably comprise at least 5 bases
    # due to the dynamic programming algorithm
    ref_dotbracket = "xxx(((((xxxxxx)))x))"
    ref_dotbracket_array = np.array(list(ref_dotbracket), dtype="U1")

    app = RNAalifoldApp(alignment)
    app.set_constraints(
        pairs=np.stack([
            np.where(ref_dotbracket_array == "(")[0],
            np.where(ref_dotbracket_array == ")")[0][::-1]
        ], axis=-1),
        unpaired = (ref_dotbracket_array == "x"),
        enforce=True
    )
    app.start()
    app.join()
    test_dotbracket = app.get_dot_bracket()

    assert test_dotbracket == ref_dotbracket.replace("x", ".")