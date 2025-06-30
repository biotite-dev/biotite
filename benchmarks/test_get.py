import pytest
import fastpdb


@pytest.mark.benchmark
def test_read(pdb_file_path):
    fastpdb.PDBFile.read(pdb_file_path)


@pytest.mark.benchmark
def test_get_coord(pdb_file):
    pdb_file.get_coord(model=1)


@pytest.mark.benchmark
def test_get_structure(pdb_file):
    pdb_file.get_structure(model=1)


@pytest.mark.benchmark
def test_get_structure_with_bonds(pdb_file):
    pdb_file.get_structure(model=1, include_bonds=True)


@pytest.mark.benchmark
def test_get_remark(pdb_file):
    pdb_file.get_remark(350)