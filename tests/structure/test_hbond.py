# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import json
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure.io import load_structure
from tests.util import data_dir


@pytest.fixture()
def stack(request):
    stack = load_structure(join(data_dir("structure"), "1l2y.bcif"))
    if request.param:
        # Use connect_via_distances, since 1l2y has invalidly bonded
        # N-terminal hydrogen atoms
        stack.bonds = struc.connect_via_distances(stack[0])
    return stack


@pytest.mark.parametrize(
    "pdb_id, use_all_models, use_bond_list",
    itertools.product(["1l2y", "1gya", "1igy"], [False, True], [False, True]),
)
def test_hbond_consistency(pdb_id, use_all_models, use_bond_list):
    """
    Compare hydrogen bond detection with MDTraj.
    """
    # Load precomputed hydrogen bond triplets from MDTraj
    with open(join(data_dir("structure"), "misc", "hbond.json")) as file:
        ref_data = json.load(file)
    key = "single_model" if not use_all_models else "all_models"
    ref_triplets = ref_data[pdb_id][key]

    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), pdb_id + ".bcif"))
    model = None if use_all_models else 1
    atoms = pdbx.get_structure(pdbx_file, model=model)

    if use_bond_list:
        if isinstance(atoms, struc.AtomArrayStack):
            ref_model = atoms[0]
        else:
            ref_model = atoms
        bonds = struc.connect_via_distances(ref_model)
        bonds = bonds.merge(struc.connect_via_residue_names(ref_model))
        atoms.bonds = bonds

    # Only consider amino acids for consistency
    # with bonded hydrogen detection in MDTraj
    atoms = atoms[..., struc.filter_amino_acids(atoms)]
    if isinstance(atoms, struc.AtomArrayStack):
        # For consistency with MDTraj 'S' cannot be acceptor element
        # https://github.com/mdtraj/mdtraj/blob/master/mdtraj/geometry/hbond.py#L365
        test_triplets, mask = struc.hbond(atoms, acceptor_elements=("O", "N"))
    else:
        test_triplets = struc.hbond(atoms, acceptor_elements=("O", "N"))

    # MDTraj and Biotite may use different order -> use set for comparison
    assert set([tuple(tri) for tri in test_triplets]) == set(
        [tuple(tri) for tri in ref_triplets]
    )


# Ignore warning about missing BondList, as this is intended
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("stack", [False, True], indirect=["stack"])
def test_hbond_same_res(stack):
    """
    Check if hydrogen bonds in the same residue are detected.
    At least one of such bonds is present in 1L2Y (1ASN with N-terminus)
    (model 2).
    """
    selection = stack.res_id == 1
    # Focus on second model
    array = stack[1]
    triplets = struc.hbond(array, selection, selection)
    assert len(triplets) == 1


# Ignore warning about missing BondList, as this is intended
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("stack", [False, True], indirect=["stack"])
def test_hbond_total_count(stack):
    """
    With the standart Baker & Hubbard criterion,
    1l2y should have 28 hydrogen bonds with a frequency > 0.1
    (comparison with results obtained from MDTraj)
    """
    triplets, mask = struc.hbond(stack)
    freq = struc.hbond_frequency(mask)

    assert len(freq[freq >= 0.1]) == 28


# Ignore warning about missing BondList, as this is intended
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("stack", [False, True], indirect=["stack"])
def test_hbond_with_selections(stack):
    """
    When selection1 and selection2 is defined, no hydrogen bonds outside
    of this boundary should be found. Also, hbond should respect the
    selection type.
    """
    selection1 = (stack.res_id == 3) & (stack.atom_name == "O")  # 3TYR BB Ox
    selection2 = stack.res_id == 7

    # backbone hbond should be found if selection1/2 type is both
    triplets, mask = struc.hbond(stack, selection1, selection2, selection1_type="both")
    assert len(triplets) == 1
    assert triplets[0][0] == 116
    assert triplets[0][2] == 38

    # backbone hbond should be found if selection1 is acceptor and
    # selection2 is donor
    triplets, mask = struc.hbond(
        stack, selection1, selection2, selection1_type="acceptor"
    )
    assert len(triplets) == 1
    assert triplets[0][0] == 116
    assert triplets[0][2] == 38

    # no hbond should be found,
    # because the backbone oxygen cannot be a donor
    triplets, mask = struc.hbond(stack, selection1, selection2, selection1_type="donor")
    assert len(triplets) == 0


# Ignore warning about missing BondList, as this is intended
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("stack", [False, True], indirect=["stack"])
def test_hbond_single_selection(stack):
    """
    If only selection1 or selection2 is defined, hbond should run
    against all other atoms as the other selection.
    """
    selection = (stack.res_id == 2) & (stack.atom_name == "O")  # 2LEU BB Ox
    triplets, mask = struc.hbond(stack, selection1=selection)
    assert len(triplets) == 2

    triplets, mask = struc.hbond(stack, selection2=selection)
    assert len(triplets) == 2


def test_hbond_frequency():
    mask = np.array(
        [
            [True, True, True, True, True],  # 1.0
            [False, False, False, False, False],  # 0.0
            [False, False, False, True, True],  # 0.4
        ]
    ).T
    freq = struc.hbond_frequency(mask)
    assert not np.isin(False, np.isclose(freq, np.array([1.0, 0.0, 0.4])))


# Ignore warning about missing BondList
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("translation_vector", [(10, 20, 30), (-5, 3, 18)])
def test_hbond_periodicity(translation_vector):
    """
    Test whether hydrogen bond identification uses periodic boundary
    conditions correctly.
    For this purpose a structure containing water is loaded and the
    hydrogen bonds are identified.
    Then the position of the periodic boundary is changed and it is
    expected that all hydrogen bonds are still the same
    """
    stack = load_structure(join(data_dir("structure"), "waterbox.gro"))
    array = stack[0]
    ref_hbonds = struc.hbond(array, periodic=True)
    # Put H-bond triplets into as stack for faster comparison with
    # set for moved atoms
    ref_hbonds = set([tuple(triplet) for triplet in ref_hbonds])
    # Move system and put back into box
    # -> Equal to move of periodic boundary
    array = struc.translate(array, translation_vector)
    array.coord = struc.move_inside_box(array.coord, array.box)
    hbonds = struc.hbond(array, periodic=True)
    hbonds = set([tuple(triplet) for triplet in hbonds])
    assert ref_hbonds == hbonds
