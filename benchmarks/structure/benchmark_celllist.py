from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(scope="module")
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1gya.bcif")
    return pdbx.get_structure(pdbx_file, model=1)


@pytest.fixture(scope="module")
def cell_list(atoms):
    return struc.CellList(atoms, 5.0)


@pytest.mark.benchmark
def benchmark_cell_list_creation(atoms):
    """
    Create a cell list for a structure.
    """
    struc.CellList(atoms, 5.0)


@pytest.mark.benchmark
def benchmark_cell_list_compute_contacts(cell_list, atoms):
    """
    Find all contacts in a structure using an existing cell list.
    """
    cell_list.get_atoms(atoms.coord, 5.0)
