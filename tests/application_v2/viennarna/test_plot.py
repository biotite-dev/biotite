# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
from biotite.application_v2.viennarna import PlotApp
from biotite.structure.dotbracket import base_pairs_from_dot_bracket
from tests.util import is_not_installed


@pytest.mark.skipif(is_not_installed("RNAplot"), reason="'RNAplot' is not installed")
def test_plot():
    """
    ``RNAplot`` places base-paired symbols close together: the distance
    between paired symbols is much smaller than the maximum distance
    between any two symbols in the plot.
    """
    # Base-paired symbols must be at least this factor closer to each other than the
    # two farthest apart symbols in the plot
    BP_CLOSENESS = 5

    dot_bracket = "((((((....))))))"
    coordinates = PlotApp().run(dot_bracket).result()
    assert coordinates.shape == (len(dot_bracket), 2)

    base_pairs = base_pairs_from_dot_bracket(dot_bracket)
    base_pair_distances = np.linalg.norm(
        coordinates[base_pairs[:, 0]] - coordinates[base_pairs[:, 1]], axis=1
    )
    max_distance = np.linalg.norm(
        coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=-1
    ).max()

    assert np.all(base_pair_distances * BP_CLOSENESS < max_distance)
