# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
from biotite.application_v2.viennarna.util import build_constraint_string


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        # No constraints
        ({}, "............"),
        # A constrained base pair becomes matching round brackets
        ({"pairs": np.array([[2, 9], [3, 8]])}, "..((....)).."),
        # Paired marker
        ({"paired": np.array([4, 5])}, "....||......"),
        # Unpaired marker
        ({"unpaired": np.array([0, 11])}, "x..........x"),
        # Downstream and upstream markers
        (
            {"downstream": np.array([1]), "upstream": np.array([10])},
            ".<........>.",
        ),
    ],
)
def test_build_constraint_string(kwargs, expected):
    """
    Check the result of `build_constraint_string()` against known examples.
    """
    assert build_constraint_string(12, **kwargs) == expected


def test_build_constraint_string_conflict():
    """
    Two constraints at the same position conflict and raise an error.
    """
    with pytest.raises(ValueError):
        build_constraint_string(12, paired=np.array([4]), unpaired=np.array([4]))
