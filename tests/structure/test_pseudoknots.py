# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import json
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
    Check the output of :func:`pseudoknots()`.
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
        assert np.max(optimal_solution) == 2

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
    Load sample base pair arrays and reference solutions from file.
    """
    # Base pairs as numpy array (input for `pseudoknots()`)
    with open(
        join(data_dir("structure"), "pseudoknots", f"{name}_knotted.json"),
        "r"
    ) as f:
        basepairs = np.array(json.load(f))
    # List of solutions (set of tuples)
    with open(
        join(data_dir("structure"), "pseudoknots", f"{name}_unknotted.json"),
        "rb"
    ) as f:
        solutions = json.load(f)
    for i, solution in enumerate(solutions):
        solutions[i] = set([tuple(pair) for pair in solution])
    return basepairs, solutions

@pytest.mark.parametrize("name", [f"test{x}" for x in range(21)])
def test_pseudoknot_removal(name):
    """
    Test the implementation of the dynamic programming algorithm
    referenced in :func:`pseudoknots()` against the original
    implementation.

    The reference solutions were created with the following tool:
    https://www.ibi.vu.nl/programs/k2nwww/

    The original purpose was to remove pseudoknots. Thus, the reference
    solutions contain unknotted basepairs.
    """
    # Get base pairs and reference solutions
    basepairs, reference_solutions = load_test(name)

    # Calculate solutions from the base pairs
    raw_solutions = struc.pseudoknots(basepairs, max_pseudoknot_order=0)

    # The number of solutions calculated
    solutions_count = 0

    # Verify that each solution is in the reference solutions
    for raw_solution in raw_solutions:
        solution = set()
        for basepair, order in zip(basepairs, raw_solution):
            if order == -1:
                continue
            solution.add(tuple(sorted(basepair)))
        solutions_count += 1
        assert solution in reference_solutions

    # Verify that the number of solutions matches the reference
    assert len(reference_solutions) == solutions_count

@pytest.mark.parametrize("seed", range(10))
def test_pseudoknot_orders(seed):
    """
    Generate a random set of basepairs. Assert that increasing
    pseudoknot orders contain less or equal base pairs. Furthermore,
    assert that each individual order only contains unknotted base
    pairs.
    """
    # Generate Random set of basepairs
    np.random.seed(seed)
    bases = range(100)
    basepairs = np.random.choice(bases, size=(20, 2), replace=False)

    # Get pseudoknot order for each basepair
    solutions = struc.pseudoknots(basepairs)

    # Iterate through the solutions
    for solution in solutions:
        # Number of base pairs in the previous order
        previous_order = -1
        for order in range(np.max(solution)+1):
            # Ensure that the base pairs of the same order are unknotted
            assert (struc.pseudoknots(basepairs[solution == order]) == 0).all()

            # Number of base pairs in the current order
            this_order = len(solution[solution == order])
            # Ensure that that higher orders contain less or equal base
            # pairs than lower orders
            if previous_order != -1:
                assert this_order <= previous_order
            previous_order = this_order

def test_empty_base_pairs():
    """
    Assert than an empty array of base pairs generates an empty array of
    pseudoknot orders. 
    """
    assert struc.pseudoknots([]).shape == (1,0)