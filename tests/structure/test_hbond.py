# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import biotite.structure as struc
from biotite.structure.io import load_structure
from .util import data_dir


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
    # with vectorization
    stack = load_structure(join(data_dir, "1l2y.mmtf"))
    triplets, mask = struc.hbond(stack, vectorized=True)
    freq = struc.hbond_frequency(mask)

    assert len(freq[freq >= 0.1]) == 28

    # without vectorization
    triplets, mask = struc.hbond(stack, vectorized=False)
    freq = struc.hbond_frequency(mask)

    assert len(freq[freq >= 0.1]) == 28


def test_hbond_with_selections():
    """
    When selection1 and selection2 is defined, no hydrogen bonds outside
    of this boundary should be found. Also, hbond should respect the selection
    type
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

    # backbone hbond should be found if selection1 is acceptor and selection2
    # is donor
    triplets, mask = struc.hbond(stack, selection1, selection2,
                                 selection1_type='acceptor')
    assert len(triplets) == 1
    assert triplets[0][0] == 116
    assert triplets[0][2] == 38

    # no hbond should be found because the backbone oxygen cannot be a donor
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


def test_is_hbond():
    from biotite.structure.hbond import _is_hbond as is_hbond
    hbond_coords_valid = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0]  # acceptor
    ])

    hbond_coords_wrong_angle = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [2.0, 2.0, 0.0]   # acceptor
    ])


    hbond_coords_wrong_distance = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [4.0, 3.0, 0.0]  # acceptor
    ])

    assert is_hbond(*hbond_coords_valid)
    assert not is_hbond(*hbond_coords_wrong_angle)
    assert not is_hbond(*hbond_coords_wrong_distance)

def test_is_hbond_multiple():
    from biotite.structure.hbond import _is_hbond as is_hbond
    hbonds_valid = np.array([
    # first model
    [
        # first bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0],   # acceptor
        # second bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0],  # acceptor
    ],
    # second model
    [
        # first bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0],   # acceptor
        # second bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0],  # acceptor
    ]])

    hbonds_invalid = np.array([
    # first model
    [
        # first bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0],   # acceptor
        # second bond
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 1.0, 0.0],  # acceptor
    ]])

    assert is_hbond(hbonds_valid[:, 0::3],
                          hbonds_valid[:, 1::3],
                          hbonds_valid[:, 2::3]).sum() == 4

    assert is_hbond(hbonds_invalid[:, 0::3],
                          hbonds_invalid[:, 1::3],
                          hbonds_invalid[:, 2::3]).sum() == 1


