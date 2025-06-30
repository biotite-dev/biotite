import pytest
import fastpdb


@pytest.fixture
def atoms(pdb_file):
    return pdb_file.get_structure(model=1, include_bonds=True)


@pytest.fixture
def empty_pdb_file():
    return fastpdb.PDBFile()


@pytest.mark.benchmark
def test_set_structure(atoms, empty_pdb_file):
    atoms.bonds = None
    empty_pdb_file.set_structure(atoms)


@pytest.mark.benchmark
def test_set_structure_with_bonds(atoms, empty_pdb_file):
    empty_pdb_file.set_structure(atoms)