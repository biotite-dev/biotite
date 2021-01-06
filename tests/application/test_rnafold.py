# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
from biotite.application.viennarna import RNAfoldApp
from ..util import is_not_installed


@pytest.fixture
def sample_app():
    """
    Provide a `RNAfoldApp` object, where RNAfold has been executed for a
    sample sequence.
    """
    sequence = seq.NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
    app = RNAfoldApp(sequence)
    app.start()
    app.join()
    return app


@pytest.mark.skipif(
    is_not_installed("RNAfold"), reason="RNAfold is not installed"
)
def test_get_dot_bracket(sample_app):
    assert sample_app.get_dot_bracket() ==  "(((.((((.......)).)))))...."


@pytest.mark.skipif(
    is_not_installed("RNAfold"), reason="RNAfold is not installed"
)
def test_get_mfe(sample_app):
    assert sample_app.get_mfe() == -1.3

@pytest.mark.skipif(
    is_not_installed("RNAfold"), reason="RNAfold is not installed"
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