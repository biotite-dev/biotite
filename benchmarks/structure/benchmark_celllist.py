from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1gya.bcif")
    return pdbx.get_structure(pdbx_file, model=1)


def benchmark_cell_list(atoms):
    """
    Find all contacts in a structure using a cell list.
    """
    cell_list = struc.CellList(atoms, 5.0)
    cell_list.get_atoms(atoms.coord, 5.0)
