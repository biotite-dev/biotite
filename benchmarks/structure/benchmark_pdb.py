from pathlib import Path
import pytest
import biotite.structure.io.pdb as pdb
from biotite.structure.info.ccd import get_ccd


@pytest.fixture(autouse=True, scope="session")
def load_ccd():
    """
    Ensure that the CCD is already loaded to avoid biasing tests with its loading time.
    """
    get_ccd()


@pytest.fixture
def pdb_file_path():
    return Path(__file__).parents[1] / "tests" / "data" / "1aki.pdb"


@pytest.fixture
def empty_pdb_file():
    return pdb.PDBFile()


@pytest.fixture
def pdb_file(pdb_file_path):
    return pdb.PDBFile.read(pdb_file_path)


@pytest.mark.benchmark
def benchmark_read(pdb_file_path):
    pdb.PDBFile.read(pdb_file_path)


@pytest.mark.benchmark
def benchmark_get_coord(pdb_file):
    pdb_file.get_coord(model=1)


@pytest.mark.benchmark
def benchmark_get_structure(pdb_file):
    pdb_file.get_structure(model=1)


@pytest.mark.benchmark
def benchmark_get_structure_with_bonds(pdb_file):
    pdb_file.get_structure(model=1, include_bonds=True)


@pytest.mark.benchmark
def benchmark_get_remark(pdb_file):
    pdb_file.get_remark(350)


@pytest.fixture
def atoms(pdb_file):
    return pdb_file.get_structure(model=1, include_bonds=True)


@pytest.mark.benchmark
def benchmark_set_structure(atoms, empty_pdb_file):
    atoms.bonds = None
    empty_pdb_file.set_structure(atoms)


@pytest.mark.benchmark
def benchmark_set_structure_with_bonds(atoms, empty_pdb_file):
    empty_pdb_file.set_structure(atoms)
