from os.path import join
import numpy as np
import pytest
import biotite.interface.pymol as pymol_interface
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

SAMPLE_COUNT = 20


@pytest.mark.parametrize("random_seed", [i for i in range(SAMPLE_COUNT)])
def test_select(random_seed):
    """
    Check if the PyMOL selection created from a boolean mask is equivalent to the
    boolean mask.
    Use the B factor as indicator for masked atoms.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    array = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)

    # Use B factor as indicator if the selection was correctly applied
    array.set_annotation("b_factor", np.zeros(array.array_length()))
    pymol_object = pymol_interface.PyMOLObject.from_structure(array)

    np.random.seed(random_seed)
    ref_mask = np.random.choice([False, True], array.array_length())

    # The method that is actually tested
    test_selection = pymol_object.where(ref_mask)
    # Set B factor of all masked atoms to 1
    pymol_interface.cmd.alter(test_selection, "b=1.0")
    test_b_factor = pymol_object.to_structure(state=1).b_factor
    # Get the mask from the occupancy back again
    test_mask = test_b_factor == 1.0

    assert np.array_equal(test_mask, ref_mask)
