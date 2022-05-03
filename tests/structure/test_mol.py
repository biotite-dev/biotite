# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import datetime
import glob
import itertools
from os.path import join, split, splitext
from tempfile import TemporaryFile
import numpy as np
import pytest
import biotite.structure.info as info
import biotite.structure.io.mol as mol
from biotite.structure.bonds import BondType
from biotite.structure.io.ctab import BOND_TYPE_MAPPING_REV
from ..util import data_dir


def test_header_conversion():
    """
    Write known example data to the header of a MOL file and expect
    to retrieve the same information when reading the file again.
    """
    ref_header = (
        "TestMol", "JD", "Biotite",
        datetime.datetime.now().replace(second=0, microsecond=0),
        "3D", "Lorem", "Ipsum", "123", "Lorem ipsum dolor sit amet"
    )

    mol_file = mol.MOLFile()
    mol_file.set_header(*ref_header)
    print(mol_file)
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    temp.seek(0)
    mol_file = mol.MOLFile.read(temp)
    test_header = mol_file.get_header()
    temp.close()

    assert test_header == ref_header


@pytest.mark.parametrize(
    "path, omit_charge",
    itertools.product(
        glob.glob(join(data_dir("structure"), "molecules", "*.sdf")),
        [False, True]
    )
)
def test_structure_conversion(path, omit_charge):
    """
    After reading a MOL file, writing the structure back to a new file
    and reading it again should give the same structure.

    In this case an SDF file is used, but it is compatible with the
    MOL format.
    """
    mol_file = mol.MOLFile.read(path)
    ref_atoms = mol.get_structure(mol_file)
    print(ref_atoms.charge)
    if omit_charge:
        ref_atoms.del_annotation("charge")

    mol_file = mol.MOLFile()
    mol.set_structure(mol_file, ref_atoms)
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    temp.seek(0)
    mol_file = mol.MOLFile.read(temp)
    test_atoms = mol.get_structure(mol_file)
    if omit_charge:
        assert np.all(test_atoms.charge == 0)
        test_atoms.del_annotation("charge")
    temp.close()

    assert test_atoms == ref_atoms

    # Check bond type fallback in MolFile.set_structure
    # Update last bond type in ref_atoms with a quadruple bond type
    ref_atoms.bonds.add_bond(0, 1, BondType.QUADRUPLE)
    updated_bond = ref_atoms.bonds.as_array()[
        np.all(ref_atoms.bonds.as_array()[:,[0,1]] == [0,1], axis=1)
    ]
    assert updated_bond.tolist()[0][2] == BondType.QUADRUPLE
    test_mol_file = mol.MOLFile()
    mol.set_structure(test_mol_file, ref_atoms)
    # test bond type fallback to BondType.ANY value (8) in
    # MolFile.set_structure to format mol_file.lines
    updated_line = [
        mol_line
        for mol_line in test_mol_file.lines if mol_line.startswith('  1  2  ')
    ].pop()
    assert int(updated_line[8]) == \
        BOND_TYPE_MAPPING_REV[BondType.ANY]
    # test bond type fallback to BondType.SINGLE value (1) in
    # MolFile.set_structure to format mol_file.lines
    mol.set_structure(test_mol_file, ref_atoms,
                      default_bond_type=BondType.SINGLE)
    updated_line = [
        mol_line
        for mol_line in test_mol_file.lines if mol_line.startswith('  1  2  ')
    ].pop()
    assert int(updated_line[8]) == \
        BOND_TYPE_MAPPING_REV[BondType.SINGLE]


@pytest.mark.parametrize(
    "path", glob.glob(join(data_dir("structure"), "molecules", "*.sdf")),
)
def test_pdbx_consistency(path):
    """
    Check if the structure parsed from a MOL file is equal to the same
    structure read from the *Chemical Component Dictionary* in PDBx
    format.

    In this case an SDF file is used, but it is compatible with the
    MOL format.
    """
    mol_name = split(splitext(path)[0])[1]
    ref_atoms = info.residue(mol_name)
    # The CCD contains information about aromatic bond types,
    # but the SDF test files do not
    ref_atoms.bonds.remove_aromaticity()

    mol_file = mol.MOLFile.read(path)
    test_atoms = mol_file.get_structure()

    assert test_atoms.coord.shape == ref_atoms.coord.shape
    assert test_atoms.coord.flatten().tolist() \
        == ref_atoms.coord.flatten().tolist()
    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
    assert set(tuple(bond) for bond in test_atoms.bonds.as_array()) \
        == set(tuple(bond) for bond in  ref_atoms.bonds.as_array())
