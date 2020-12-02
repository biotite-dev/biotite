# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
import pickle as pkl
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

def load_test(name):
    """
    Load sample input basepair arrays and their solutions according to
    the reference algorithm which is located at:
    https://www.ibi.vu.nl/programs/k2nwww/
    """
    # Basepair numpy array (input for `pseudoknots()`)
    with open(
        join(data_dir("structure"), "pseudoknots", f"{name}_knotted.pkl"),
        "rb"
    ) as f:
        basepairs = pkl.load(f)
    # List of sulutions (set of tuples)
    with open(
        join(data_dir("structure"), "pseudoknots", f"{name}_unknotted.pkl"),
        "rb"
    ) as f:
        solutions = pkl.load(f)
    return basepairs, solutions

@pytest.mark.parametrize("name", [f"test{x}" for x in range(10)])
def test_pseudoknot_removal(name):
    basepairs, reference_solutions = load_test(name)

    reference_solutions_set = set()
    for reference_solution in reference_solutions:
        reference_solutions_set.add(frozenset(reference_solution))
    reference_solutions = reference_solutions_set

    raw_solutions = struc.pseudoknots(basepairs, max_pseudoknot_order=0)
    solutions = set()
    for raw_solution in raw_solutions:
        solution = set()
        for basepair, order in zip(basepairs, raw_solution):
            if order == -1:
                continue
            solution.add(tuple(sorted(basepair)))
        assert solution in reference_solutions
        solutions.add(frozenset(solution))
    if name == "test3" or name == "test4":
        print(reference_solutions - solutions)
    print('')
    assert len(reference_solutions) == len(solutions)






