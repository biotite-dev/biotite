import json
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def nuc_sample_array():
    """
    Sample structure for pseudoknot detection.
    """
    return pdbx.get_structure(
        pdbx.CIFFile.read(data_dir("structure") / "pdb" / "4p5j.cif"),
        model=1,
        # The expected values refer to the author fields
        use_author_fields=True,
    )


def test_pseudoknots(nuc_sample_array):
    """
    Check the output of :func:`pseudoknots()`.
    """
    # Known base pairs with pseudoknot-order = 1:
    pseudoknot_order_one = [{2, 74}, {8, 52}, {58, 72}, {59, 71}, {60, 70}]
    # Known base pairs that can be of order one, two or three
    pseudoknot_order_one_to_three = [{9, 48}, {10, 49}, {12, 50}]
    # Known base pairs, where only one of them is part of a pseudoknot:
    # Either the first one is of order two or the second one is of order one
    pseudoknot_alternatives = ({63, 83}, {82, 84})

    base_pairs = struc.base_pairs(nuc_sample_array)
    pseudoknot_order = struc.pseudoknots(base_pairs)
    res_ids = [set(base_pair) for base_pair in nuc_sample_array[base_pairs].res_id]

    # Sample structure should have twelve optimal solutions with default
    # scoring parameters
    assert len(pseudoknot_order) == 12
    assert len(np.unique(pseudoknot_order, axis=0)) == 12

    for optimal_solution in pseudoknot_order:
        assert len(base_pairs) == len(optimal_solution)

        # Assert that the each base pair has the right pseudoknot order
        for pair in pseudoknot_order_one:
            assert optimal_solution[res_ids.index(pair)] == 1
        assert sorted(
            optimal_solution[res_ids.index(pair)]
            for pair in pseudoknot_order_one_to_three
        ) == [1, 2, 3]
        assert tuple(
            optimal_solution[res_ids.index(pair)] for pair in pseudoknot_alternatives
        ) in [(2, 0), (0, 1)]
        # Assert that all other base pairs are not part of a pseudoknot
        assert np.count_nonzero(optimal_solution) == (
            len(pseudoknot_order_one) + len(pseudoknot_order_one_to_three) + 1
        )


def load_test(name):
    """
    Load sample base pair arrays and reference solutions from file.
    """
    # Base pairs as numpy array (input for `pseudoknots()`)
    with open(data_dir("structure") / "pseudoknots" / f"{name}_knotted.json", "r") as f:
        basepairs = np.array(json.load(f))
    # List of solutions (set of tuples)
    with open(
        data_dir("structure") / "pseudoknots" / f"{name}_unknotted.json", "rb"
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
    rng = np.random.default_rng(seed)
    bases = range(100)
    basepairs = rng.choice(bases, size=(20, 2), replace=False)

    # Get pseudoknot order for each basepair
    solutions = struc.pseudoknots(basepairs)

    # Iterate through the solutions
    for solution in solutions:
        # Number of base pairs in the previous order
        previous_order = -1
        for order in range(np.max(solution) + 1):
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
    assert struc.pseudoknots([]).shape == (1, 0)
