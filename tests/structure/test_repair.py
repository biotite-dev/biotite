# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def single_chain():
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "1l2y.bcif")
    return pdbx.get_structure(pdbx_file, model=1)


@pytest.fixture
def multi_chain(single_chain):
    other_chain = single_chain.copy()
    other_chain.chain_id[:] = "B"
    return single_chain + other_chain


@pytest.mark.parametrize("restart_each_chain", [False, True])
def test_create_continuous_res_ids(multi_chain, restart_each_chain):
    """
    Check if continuous residue IDs are restored, after removing a
    residue.
    """
    # Remove a residue
    multi_chain = multi_chain[multi_chain.res_id != 5]
    # Restore continuity
    multi_chain.res_id = struc.create_continuous_res_ids(
        multi_chain, restart_each_chain=restart_each_chain
    )

    test_res_ids, _ = struc.get_residues(multi_chain)
    if restart_each_chain:
        assert (
            test_res_ids.tolist()
            == np.concatenate([np.arange(len(test_res_ids) // 2) + 1] * 2).tolist()
        )
    else:
        assert test_res_ids.tolist() == (np.arange(len(test_res_ids)) + 1).tolist()


@pytest.mark.parametrize("as_atom_array", [False, True])
@pytest.mark.parametrize(
    ["elements", "expected"],
    [
        (["C", "C", "O"], ["C1", "C2", "O1"]),
        # Single atoms are named after their element
        (["CL"], ["CL"]),
        ([], []),
    ],
)
def test_create_atom_names(elements, expected, as_atom_array):
    """
    Check if atom names are correctly enumerated for both `AtomArray`
    inputs and plain array-like inputs of elements.
    """
    if as_atom_array:
        atoms = struc.AtomArray(len(elements))
        atoms.element = np.array(elements, dtype="U6")
        atoms.coord[:] = 0.0
        input_value = atoms
    else:
        input_value = elements

    assert struc.create_atom_names(input_value).tolist() == expected


@pytest.mark.filterwarnings("ignore:Could not infer element")
@pytest.mark.parametrize(
    ["name", "expected"],
    [
        ("CA", "C"),
        ("C", "C"),
        ("CB", "C"),
        ("OD1", "O"),
        ("HD21", "H"),
        ("1H", "H"),
        # ("CL", "CL"),  # This is an edge case where inference is difficult
        ("HE", "H"),
        ("SD", "S"),
        ("NA", "N"),
        ("NX", "N"),
        ("BE", "BE"),
        ("BEA", "BE"),
        ("K", "K"),
        ("KA", "K"),
        ("QWERT", ""),
    ],
)
def test_infer_elements(name, expected):
    """
    Check if elements are correctly guessed based on known examples.
    """
    assert struc.infer_elements([name])[0] == expected
