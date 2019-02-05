# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as strucinfo
from biotite.structure.io import load_structure, save_structure
import biotite.structure.io.mmtf as mmtf
from .util import data_dir


def test_mass():
    """
    Test whether the mass of a residue is the same as the sum of the
    masses of its contained atoms.
    """
    array = load_structure(join(data_dir, "1l2y.mmtf"))[0]
    _, res_names = struc.get_residues(array)
    water_mass = strucinfo.mass("H") * 2 + strucinfo.mass("O") 
    # Mass of water must be subtracted
    masses = [strucinfo.mass(res_name) - water_mass for res_name in res_names]
    # C-terminus normally has additional oxygen atom
    masses[-1] += strucinfo.mass("O")
    ref_masses = [strucinfo.mass(res) for res in struc.residue_iter(array)]
    # Up to three additional/missing hydrogens are allowed
    # (protonation state)
    mass_diff = np.abs(np.array(
        [mass - ref_mass for mass, ref_mass in zip(masses, ref_masses)]
    ))
    assert (mass_diff // strucinfo.mass("H") <= 3).all()
    assert np.allclose((mass_diff % strucinfo.mass("H")), 0, atol=5e-3)


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.mmtf")))
def test_bonds(path):
    """
    Test whether the bond data is consistent with the content of MMTF
    files.
    """
    bond_data = strucinfo.get_bond_dataset()
    mmtf_file = mmtf.MMTFFile()
    mmtf_file.read(path)
    for group in mmtf_file["groupList"]:
        group_name = group["groupName"]
        atom_names = group["atomNameList"]
        bond_indices = group["bondAtomList"]
        bond_orders = group["bondOrderList"]
        for i in range(0, len(bond_indices), 2):
            atom1 = atom_names[bond_indices[i]]
            atom2 = atom_names[bond_indices[i+1]]
            order = bond_orders[i//2]
            assert strucinfo.get_bond_order(group_name, atom1, atom2) == order
            assert frozenset((atom1, atom2)) \
                   in strucinfo.get_bonds_for_residue(group_name)
            assert frozenset((atom1, atom2)) \
                   in bond_data[group_name]

