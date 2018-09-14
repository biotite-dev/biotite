# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite
import biotite.structure as struc
from biotite.structure.io import load_structure, save_structure
from .util import data_dir


# Ignore warning about dummy unit cell vector
@pytest.mark.filterwarnings("ignore")
@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("pdb_id", ["1l2y", "1gya", "1igy"])
def test_hbond_structure(pdb_id):
    file_name = join(data_dir, pdb_id+".mmtf")
    
    array = load_structure(file_name)
    if isinstance(array, struc.AtomArrayStack):
        triplets, mask = struc.hbond(array)
    else:
        triplets = struc.hbond(array)
    
    # Save to new pdb file for consistent treatment of inscode/altloc
    # im MDTraj
    file_name = biotite.temp_file("pdb")
    save_structure(file_name, array)
    
    # Compare with MDTraj
    import mdtraj
    traj = mdtraj.load(file_name)
    triplets_ref = mdtraj.baker_hubbard(
        traj, freq=0, exclude_water=False, periodic=False
    )

    # Both packages may use different order
    # -> use set for comparison
    triplets_set = set([tuple(tri) for tri in triplets])
    triplets_ref_set = set([tuple(tri) for tri in triplets_ref])

    ###
    import biotite.structure.io.pdbx as pdbx
    file = pdbx.PDBxFile()
    file.read(join(data_dir, pdb_id+".cif"))
    array = pdbx.get_structure(file, extra_fields=["atom_id"], model=1)
    print(traj.n_atoms)
    print(array.array_length())
    id_diff = np.diff(array.atom_id)
    print(array[2536:2539])
    print(array.atom_id[2536:2539])
    print(np.where(id_diff != 1))
    print()
    print()
    try:
        for i1, i2, i3 in [
            (5940, 5946, 6081),
            (2205, 2213, 2829),
            (9022, 9028, 8999),
            (10059, 10066, 9984),
            (10159, 10163, 10157),
            (10687, 10694, 10685),
            (11527, 11531, 11516)]:
                print(array[[i1, i2, i3]])
                print(struc.distance(array[i2], array[i3]))
                print(np.rad2deg(struc.angle(array[i1], array[i2], array[i3])))
                print()
    except:
        pass
    ###
    assert triplets_set == triplets_ref_set


def test_hbond_same_res():
    """
    Check if hydrogen bonds in the same residue are detected.
    At least one of such bonds is present in 1L2Y (1ASN with N-terminus)
    (model 2).
    """
    stack = load_structure(join(data_dir, "1l2y.mmtf"))
    selection = stack.res_id == 1
    # Focus on second model
    array = stack[1]
    triplets = struc.hbond(array, selection, selection)
    assert len(triplets) == 1


def test_hbond_total_count():
    """
    With the standart Baker & Hubbard criterion,
    1l2y should have 28 hydrogen bonds with a frequency > 0.1
    (comparision with MDTraj results)
    """
    stack = load_structure(join(data_dir, "1l2y.mmtf"))
    triplets, mask = struc.hbond(stack)
    freq = struc.hbond_frequency(mask)

    assert len(freq[freq >= 0.1]) == 28


def test_hbond_with_selections():
    """
    When selection1 and selection2 is defined, no hydrogen bonds outside
    of this boundary should be found. Also, hbond should respect the
    selection type.
    """
    stack = load_structure(join(data_dir, "1l2y.mmtf"))
    selection1 = (stack.res_id == 3) & (stack.atom_name == 'O')  # 3TYR BB Ox
    selection2 = stack.res_id == 7

    # backbone hbond should be found if selection1/2 type is both
    triplets, mask = struc.hbond(stack, selection1, selection2,
                                 selection1_type='both')
    assert len(triplets) == 1
    assert triplets[0][0] == 116
    assert triplets[0][2] == 38

    # backbone hbond should be found if selection1 is acceptor and
    # selection2 is donor
    triplets, mask = struc.hbond(stack, selection1, selection2,
                                 selection1_type='acceptor')
    assert len(triplets) == 1
    assert triplets[0][0] == 116
    assert triplets[0][2] == 38

    # no hbond should be found,
    # because the backbone oxygen cannot be a donor
    triplets, mask = struc.hbond(stack, selection1, selection2,
                                 selection1_type='donor')
    assert len(triplets) == 0


def test_hbond_single_selection():
    """
    If only selection1 or selection2 is defined, hbond should run
    against all other atoms as the other selection
    """
    stack = load_structure(join(data_dir, "1l2y.mmtf"))
    selection = (stack.res_id == 2) & (stack.atom_name == 'O')  # 2LEU BB Ox
    triplets, mask = struc.hbond(stack, selection1=selection)
    assert len(triplets) == 2

    triplets, mask = struc.hbond(stack, selection2=selection)
    assert len(triplets) == 2


def test_hbond_frequency():
    mask = np.array([
        [True, True, True, True, True], # 1.0
        [False, False, False, False, False], # 0.0
        [False, False, False, True, True] # 0.4
    ]).T
    freq = struc.hbond_frequency(mask)
    assert not np.isin(False, np.isclose(freq, np.array([1.0, 0.0, 0.4])))