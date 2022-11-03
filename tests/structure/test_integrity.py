# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.npz as npz
import numpy as np
from os.path import join
from ..util import data_dir
import pytest


@pytest.fixture
def sample_array():
    file = npz.NpzFile.read(join(data_dir("structure"), "1l2y.npz"))
    return file.get_structure()[0]

@pytest.fixture
def gapped_sample_array(sample_array):
    atom_ids = np.arange(1, sample_array.shape[0]+1)
    sample_array.add_annotation("atom_id", dtype=int)
    sample_array.atom_id = atom_ids
    sample_array = sample_array[sample_array.res_id != 5]
    sample_array = sample_array[(sample_array.res_id != 9) |
                                (sample_array.atom_name != "N")]
    return sample_array

@pytest.fixture
def duplicate_sample_array(sample_array):
    sample_array[42] = sample_array[10]
    sample_array[234] = sample_array[123]
    return sample_array

def test_atom_id_continuity_check(gapped_sample_array):
    discon = struc.check_atom_id_continuity(gapped_sample_array)
    discon_array = gapped_sample_array[discon]
    assert discon_array.atom_id.tolist() == [93, 159]

def test_res_id_continuity_check(gapped_sample_array):
    discon = struc.check_res_id_continuity(gapped_sample_array)
    discon_array = gapped_sample_array[discon]
    assert discon_array.res_id.tolist() == [6]

def test_linear_continuity_check(gapped_sample_array):
    # Take the first ASN residue and remove hydrogens
    asn = gapped_sample_array[
        (gapped_sample_array.res_id == 1) & (gapped_sample_array.element != 'H')]
    # The consecutive atom groups are
    # (1) N, CA, C, O
    # - break
    # (2) CB, CG, OD1
    # - break
    # (3) ND2
    # => Indices must be 4 and 7
    discon = struc.check_linear_continuity(asn)
    assert discon.tolist() == [4, 7]

def test_bond_continuity_check(gapped_sample_array):
    discon = struc.check_backbone_continuity(gapped_sample_array)
    discon_array = gapped_sample_array[discon]
    assert discon_array.res_id.tolist() == [6,9]

def test_duplicate_atoms_check(duplicate_sample_array):
    discon = struc.check_duplicate_atoms(duplicate_sample_array)
    assert discon.tolist() == [42,234]

def test_renum_res_ids(gapped_sample_array): 
    renumbered_array = struc.renumber_res_ids(gapped_sample_array)
    # if renumbering was successful, this should not raise
    with pytest.raises(AssertionError):
        test_res_id_continuity_check(renumbered_array)

def test_renum_atom_ids(gapped_sample_array):
    renumbered_array = struc.renumber_atom_ids(gapped_sample_array)
    # if renumbering was successful, this should not raise
    with pytest.raises(AssertionError):
        test_atom_id_continuity_check(renumbered_array)