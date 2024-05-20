# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import datetime
import glob
import itertools
from os.path import join, splitext
from tempfile import TemporaryFile
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.mol as mol
import biotite.structure.io.pdbx as pdbx
from biotite.structure.bonds import BondType
from biotite.structure.io.mol.ctab import BOND_TYPE_MAPPING_REV
from ..util import data_dir


def list_v2000_sdf_files():
    return [
        path for path
        in glob.glob(join(data_dir("structure"), "molecules", "*.sdf"))
        if not "v3000" in path
    ]

def list_v3000_sdf_files():
    return glob.glob(join(data_dir("structure"), "molecules", "*v3000.sdf"))


def toy_atom_array(n_atoms):
    atoms = struc.AtomArray(n_atoms)
    atoms.coord[:] = 1.0
    atoms.element[:] = "H"
    atoms.add_annotation("charge", dtype=int)
    atoms.charge[:] = 0
    atoms.bonds = struc.BondList(n_atoms)
    return atoms


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
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    temp.seek(0)
    mol_file = mol.MOLFile.read(temp)
    test_header = mol_file.get_header()
    temp.close()

    assert test_header == ref_header


@pytest.mark.parametrize(
    "path, version, omit_charge",
    itertools.product(
        list_v2000_sdf_files(),
        ["V2000", "V3000"],
        [False, True]
    )
)
def test_structure_conversion(path, version, omit_charge):
    """
    After reading a MOL file, writing the structure back to a new file
    and reading it again should give the same structure.

    In this case an SDF file is used, but it is compatible with the
    MOL format.
    """
    mol_file = mol.MOLFile.read(path)
    ref_atoms = mol.get_structure(mol_file)
    if omit_charge:
        ref_atoms.del_annotation("charge")

    mol_file = mol.MOLFile()
    mol.set_structure(mol_file, ref_atoms, version=version)
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


@pytest.mark.parametrize(
    "path", list_v2000_sdf_files() + list_v3000_sdf_files()
)
def test_pdbx_consistency(path):
    """
    Check if the structure parsed from a MOL file is equal to the same
    structure read in PDBx format.

    In this case an SDF file is used, but it is compatible with the
    MOL format.
    """
    # Remove '.sdf' and optional '.v3000' suffix
    cif_path = splitext(splitext(path)[0])[0] + ".cif"

    pdbx_file = pdbx.CIFFile.read(cif_path)
    ref_atoms = pdbx.get_component(pdbx_file)
    # The PDBx test files contain information about aromatic bond types,
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


@pytest.mark.parametrize("path", list_v2000_sdf_files())
def test_structure_bond_type_fallback(path):
    """
    Check if a bond with a type not supported by MOL files will be translated
    thanks to the bond type fallback in `MolFile.set_structure`
    """
    # Extract original list of bonds from an SDF file
    mol_file = mol.MOLFile.read(path)
    ref_atoms = mol.get_structure(mol_file)
    # Update one bond in `ref_atoms` with with a quadruple bond type,
    # which is not supported by MOL files and thus translates to
    # the default bond type
    ref_atoms.bonds.add_bond(0, 1, BondType.QUADRUPLE)
    updated_bond = ref_atoms.bonds.as_array()[
        np.all(ref_atoms.bonds.as_array()[:,[0,1]] == [0,1], axis=1)
    ]
    assert updated_bond.tolist()[0][2] == BondType.QUADRUPLE
    test_mol_file = mol.MOLFile()
    mol.set_structure(test_mol_file, ref_atoms)
    # Test bond type fallback to BondType.ANY value (8) in
    # MolFile.set_structure during mol_file.lines formatting
    updated_line = [
        mol_line
        for mol_line in test_mol_file.lines if mol_line.startswith('  1  2  ')
    ].pop()
    assert int(updated_line[8]) == \
        BOND_TYPE_MAPPING_REV[BondType.ANY]
    # Test bond type fallback to BondType.SINGLE value (1) in
    # MolFile.set_structure during mol_file.lines formatting
    mol.set_structure(test_mol_file, ref_atoms,
                      default_bond_type=BondType.SINGLE)
    updated_line = [
        mol_line
        for mol_line in test_mol_file.lines if mol_line.startswith('  1  2  ')
    ].pop()
    assert int(updated_line[8]) == \
        BOND_TYPE_MAPPING_REV[BondType.SINGLE]


@pytest.mark.parametrize("atom_type", ["", " ", "A ", " A"])
def test_quoted_atom_types(atom_type):
    """
    Check if V3000 MOL files can handle atom types (aka elements) with
    empty strings or whitespaces.
    """
    ref_atoms = toy_atom_array(1)
    ref_atoms.element[0] = atom_type
    mol_file = mol.MOLFile()
    mol_file.set_structure(ref_atoms, version="V3000")
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    temp.seek(0)
    mol_file = mol.MOLFile.read(temp)
    test_atoms = mol_file.get_structure()
    assert test_atoms.element[0] == atom_type
    # Also check if the rest of the structure was parsed correctly
    assert test_atoms == ref_atoms


def test_large_structure():
    """
    Check if MOL files automatically switch to V3000 format if the
    number of atoms exceeds the fixed size columns in the table.
    """
    ref_atoms = toy_atom_array(1000)
    mol_file = mol.MOLFile()
    # Let the MOL file automatically switch to V3000 format
    mol_file.set_structure(ref_atoms, version=None)
    temp = TemporaryFile("w+")
    mol_file.write(temp)

    temp.seek(0)
    mol_file = mol.MOLFile.read(temp)
    test_atoms = mol_file.get_structure()
    # Check if file is written in V3000 format
    assert "V3000" in str(mol_file)
    assert test_atoms == ref_atoms