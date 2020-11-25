# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from os.path import join
from ..util import data_dir


@pytest.fixture
def nuc_sample_array():
    """
    Sample structure.
    """
    return strucio.load_structure(join(data_dir("structure"), "4p5j.cif"))

def test_dot_bracket_from_structure(nuc_sample_array):
    """
    Check the output of ``dot_bracket_from_structure()``.
    """
    expected_output = [
        ".[(((((.[<...)))))(((((((.......)))))))...(((((]>.)..))))[[[...(((((("
        "]]].].))))))(.)",
        ".[(((((.<[...)))))(((((((.......)))))))...(((((>].)..))))[[[...(((((("
        "]]].].))))))(.)"
    ]
    output = struc.dot_bracket_from_structure(
        nuc_sample_array[struc.filter_nucleotides(nuc_sample_array)]
    )

    # Check that each solution is a correct output
    for solution in output:
        assert solution in expected_output

    # Check that each solution is unique
    unique_solutions = set(output)
    assert len(output) == len(unique_solutions)
