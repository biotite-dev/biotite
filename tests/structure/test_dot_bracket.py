# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure.info import residue
from os.path import join
from ..util import data_dir


@pytest.fixture
def nuc_sample_array():
    """
    Sample structure.
    """
    return strucio.load_structure(join(data_dir("structure"), "4p5j.cif"))

def test_dot_bracket(nuc_sample_array):
    """
    Check the output of ``dot_bracket()``.
    """
    exspected_output = [
        ".[(((((.[<...)))))(((((((.......)))))))...(((((]>.)..))))[[[...(((((("
        "]]].].))))))(.)",
        ".[(((((.<[...)))))(((((((.......)))))))...(((((>].)..))))[[[...(((((("
        "]]].].))))))(.)"
    ]
    output = struc.dot_bracket(
        nuc_sample_array[struc.filter_nucleotides(nuc_sample_array)]
    )

    for solution in output:
        assert solution in exspected_output

    unique_solutions = set(output)
    assert len(output) == len(unique_solutions)
