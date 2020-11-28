# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
from os.path import join
from ..util import data_dir


@pytest.fixture
def nuc_sample_array():
    """
    Sample structure for pseudoknot detection.
    """
    return strucio.load_structure(join(data_dir("structure"), "4p5j.cif"))

def test_pseudoknots(nuc_sample_array):
    """
    Check the output of ``pseudoknots()``.
    """
    # Known base pairs with pseudoknot-order = 1:
    pseudoknot_order_one = [{2, 74}, {58, 72}, {59, 71}, {60, 70}]
    # Known base pairs that can either be of order one or two
    pseudoknot_order_one_or_two =  [{9, 48}, {10, 49}]
    order_one_count = (
        len(pseudoknot_order_one) + (len(pseudoknot_order_one_or_two)/2)
    )
    order_two_count = len(pseudoknot_order_one_or_two)/2

    base_pairs = struc.base_pairs(nuc_sample_array)
    pseudoknot_order = struc.pseudoknots(base_pairs)

    # Sample structure should have two optimal solutions with default
    # scoring parameters
    assert len(pseudoknot_order) == 2

    for optimal_solution in pseudoknot_order:
        # Assert that the right number of pseudoknots is present for
        # each order
        assert len(base_pairs) == len(optimal_solution)
        assert np.count_nonzero(optimal_solution == 1) == order_one_count
        assert np.count_nonzero(optimal_solution == 2) == order_two_count
        assert np.amax(optimal_solution) == 2

        # Assert that the each base pair has the right pseudoknot order
        for base_pair, order in zip(
            nuc_sample_array[base_pairs].res_id, optimal_solution
        ):
            if(order == 1):
                assert (
                    set(base_pair) in pseudoknot_order_one or
                    set(base_pair) in pseudoknot_order_one_or_two
                )
            elif (order == 2):
                assert (
                    set(base_pair) in pseudoknot_order_one_or_two
                )




