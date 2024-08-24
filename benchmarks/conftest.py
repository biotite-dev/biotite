from pathlib import Path
import pytest
import fastpdb


@pytest.fixture
def pdb_file_path():
    return Path(__file__).parents[1] / "tests" / "data" / "1aki.pdb"


@pytest.fixture
def pdb_file(pdb_file_path):
    return fastpdb.PDBFile.read(pdb_file_path)