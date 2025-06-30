import pytest
from biotite.structure.info.ccd import get_ccd


@pytest.fixture(autouse=True, scope="session")
def load_ccd():
    """
    Ensure that the CCD is already loaded to avoid biasing tests with its loading time.
    """
    get_ccd()
from pathlib import Path
import pytest
import fastpdb


@pytest.fixture
def pdb_file_path():
    return Path(__file__).parents[1] / "tests" / "data" / "1aki.pdb"


@pytest.fixture
def pdb_file(pdb_file_path):
    return fastpdb.PDBFile.read(pdb_file_path)
