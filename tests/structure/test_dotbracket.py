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
    Sample structure.
    """
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "4p5j.cif")
    )
    return nuc_sample_array[struc.filter_nucleotides(nuc_sample_array)]

@pytest.fixture
def expected_output():
    """
    Output for sample structure
    """
    return [
        ".[(((((.[{...)))))(((((((.......)))))))...(((((]}.)..))))[[[...(((((("
        "]]].].))))))(.)",
        ".[(((((.{[...)))))(((((((.......)))))))...(((((}].)..))))[[[...(((((("
        "]]].].))))))(.)"
    ]

@pytest.fixture
def basepair_residue_positions():
    """
    The base pairs in the sample array by their residue postions.
    """
    return np.array(
        [[1, 73],
         [2, 17],
         [3, 16],
         [4, 15],
         [5, 14],
         [6, 13],
         [8, 47],
         [9, 48],
         [18, 38],
         [19, 37],
         [20, 36],
         [21, 35],
         [22, 34],
         [23, 33],
         [24, 32],
         [42, 56],
         [43, 55],
         [44, 54],
         [45, 53],
         [46, 50],
         [57, 71],
         [58, 70],
         [59, 69],
         [63, 80],
         [64, 79],
         [65, 78],
         [66, 77],
         [67, 76],
         [68, 75],
         [81, 83]]
    )

def verify_dot_bracket_notation(output, expected_output):
    """
    Ensure that the dot_bracket notation matches a reference.
    """
    # Check that each solution is a correct output
    for solution in output:
        assert solution in expected_output

    # Check that each solution is unique
    unique_solutions = set(output)
    assert len(output) == len(unique_solutions)

def test_dot_bracket_from_structure(nuc_sample_array, expected_output):
    """
    Check the output of ``dot_bracket_from_structure()``.
    """
    output = struc.dot_bracket_from_structure(nuc_sample_array)
    verify_dot_bracket_notation(output, expected_output)

def test_dot_bracket(basepair_residue_positions, expected_output):
    """
    Check the output of ``dot_bracket()``.
    """
    output = struc.dot_bracket(
        basepair_residue_positions, len(expected_output[0])
    )
    verify_dot_bracket_notation(output, expected_output)

def test_base_pairs_from_dot_bracket(
    basepair_residue_positions, expected_output
):
    """
    Ensure that the base pairs are correctly extracted from the
    DBL-notation
    """
    for notation in expected_output:
        computed_residue_positions = struc.base_pairs_from_dot_bracket(notation)
        assert np.all(computed_residue_positions == basepair_residue_positions)