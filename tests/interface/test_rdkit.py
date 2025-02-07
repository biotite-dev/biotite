from pathlib import Path
import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import Atom, EditableMol, Mol
from rdkit.Chem.rdchem import BondType as RDKitBondType
from rdkit.Chem.rdmolops import (
    AddHs,
    RemoveStereochemistry,
)
import biotite.interface.rdkit as rdkit_interface
import biotite.structure as struc
import biotite.structure.info as info
from biotite.interface import LossyConversionWarning
from tests.util import data_dir


def _load_smiles():
    with open(Path(data_dir("interface")) / "smiles.txt") as file:
        return file.read().splitlines()


@pytest.mark.filterwarnings("ignore:Missing coordinates.*")
@pytest.mark.filterwarnings("ignore:.*coordinates are missing.*")
@pytest.mark.filterwarnings("ignore::biotite.interface.LossyConversionWarning")
@pytest.mark.parametrize(
    "res_name", np.random.default_rng(0).choice(info.all_residues(), size=200).tolist()
)
def test_conversion_from_biotite(res_name):
    """
    Test a round trip conversion of a small molecule (single residue) from Biotite to
    RDKit and back and expect to recover the same molecule.

    Run this on randomly selected molecules from the CCD.
    """
    ref_atoms = info.residue(res_name, allow_missing_coord=True)

    mol = rdkit_interface.to_mol(ref_atoms)
    test_atoms = rdkit_interface.from_mol(mol, add_hydrogen=False)

    assert test_atoms.atom_name.tolist() == ref_atoms.atom_name.tolist()
    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
    # Some compounds in the CCD have missing coordinates
    assert np.allclose(test_atoms.coord, ref_atoms.coord, equal_nan=True)

    # There should be no undefined bonds
    assert (test_atoms.bonds.as_array()[:, 2] != struc.BondType.ANY).all()
    # Kekulization returns one of multiple resonance structures, so the returned one
    # might not be the same as the input
    # -> Only check non aromatic bonds for equality
    ref_is_aromatic = np.isin(
        ref_atoms.bonds.as_array()[:, 2],
        [
            struc.BondType.AROMATIC_SINGLE,
            struc.BondType.AROMATIC_DOUBLE,
            struc.BondType.AROMATIC_TRIPLE,
            struc.BondType.AROMATIC,
        ],
    )
    test_is_aromatic = np.isin(
        test_atoms.bonds.as_array()[:, 2],
        [
            struc.BondType.AROMATIC_SINGLE,
            struc.BondType.AROMATIC_DOUBLE,
            struc.BondType.AROMATIC_TRIPLE,
            struc.BondType.AROMATIC,
        ],
    )
    assert np.all(ref_is_aromatic == test_is_aromatic)
    # Check also the non-aromatic bonds
    assert set(
        tuple(bond) for bond in test_atoms.bonds.as_array()[~test_is_aromatic]
    ) == set(tuple(bond) for bond in ref_atoms.bonds.as_array()[~ref_is_aromatic])


def test_conversion_from_biotite_multi_model():
    """
    Same as :func:`test_conversion_from_biotite()`, but with a multi-model structure.
    """
    RES_NAME = "ALA"
    STACK_DEPTH = 1

    ref_atoms = struc.stack([info.residue(RES_NAME)] * STACK_DEPTH)

    mol = rdkit_interface.to_mol(ref_atoms)
    test_atoms = rdkit_interface.from_mol(mol)

    assert test_atoms.atom_name.tolist() == ref_atoms.atom_name.tolist()
    assert test_atoms.element.tolist() == ref_atoms.element.tolist()
    assert test_atoms.charge.tolist() == ref_atoms.charge.tolist()
    assert np.allclose(test_atoms.coord.tolist(), ref_atoms.coord.tolist())
    assert test_atoms.bonds.as_set() == ref_atoms.bonds.as_set()


@pytest.mark.parametrize("smiles", _load_smiles())
def test_conversion_from_rdkit(smiles):
    """
    Test a round trip conversion of a small molecule (single residue) from RDKit to
    Biotite and back and expect to recover the same molecule.

    Start from SMILES string to ensure that built-in functionality of RDKit is used
    to create the initial molecule.
    """
    ref_mol = MolFromSmiles(smiles)
    atoms = rdkit_interface.from_mol(ref_mol)
    test_mol = rdkit_interface.to_mol(atoms)

    # The intermediate AtomArray has explicit hydrogen atoms so add them explicitly
    # to the reference as well for fair comparison
    ref_mol = AddHs(ref_mol)
    # The intermediate AtomArray does not have stereochemistry information,
    # so this info cannot be preserved in the comparison
    RemoveStereochemistry(ref_mol)

    # RDKit does not support equality checking -> Use SMILES string as proxy
    assert MolToSmiles(test_mol) == MolToSmiles(ref_mol)


def test_kekulization():
    """
    Check if a benzene ring has alternating single and double bonds.
    """
    atoms = info.residue("BNZ")
    atoms = atoms[atoms.element != "H"]
    # Omit hydrogen for easier comparison of of aromatic bond types later on
    ref_bond_types = atoms.bonds.as_array()[:, 2]

    mol = rdkit_interface.to_mol(atoms)
    atoms = rdkit_interface.from_mol(mol, add_hydrogen=False)
    test_bond_types = atoms.bonds.as_array()[:, 2]

    assert (
        test_bond_types.tolist() == ref_bond_types.tolist()
        # There are two possible resonance structures -> swap single and double bonds
        or [
            struc.BondType.AROMATIC_SINGLE
            if btype == struc.BondType.AROMATIC_DOUBLE
            else struc.BondType.AROMATIC_SINGLE
            for btype in test_bond_types
        ]
        == ref_bond_types.tolist()
    )


def test_unmappable_bond_type():
    """
    Test that a warning is raised when a bond type cannot be mapped to Biotite.
    """
    mol = EditableMol(Mol())
    mol.AddAtom(Atom("F"))
    mol.AddAtom(Atom("F"))
    # 'HEXTUPLE' has no corresponding Biotite bond type
    mol.AddBond(0, 1, RDKitBondType.HEXTUPLE)
    mol = mol.GetMol()

    with pytest.warns(LossyConversionWarning):
        rdkit_interface.from_mol(mol)


def test_fortran_ordered_coord():
    """
    Check if :func:`to_mol()` also works with ``ndarray`` objects in *Fortran*
    contiguous order.

    Currently *RDKit* cannot handle *Fortran*-ordered arrays directly as described
    in https://github.com/rdkit/rdkit/issues/8221.
    """
    ref_atoms = info.residue("ALA", allow_missing_coord=True)
    # Bring coordinates to Fortran order
    ref_atoms.coord = np.asfortranarray(ref_atoms.coord)

    mol = rdkit_interface.to_mol(ref_atoms)
    test_atoms = rdkit_interface.from_mol(mol, add_hydrogen=False)

    assert np.allclose(test_atoms.coord, ref_atoms.coord)
