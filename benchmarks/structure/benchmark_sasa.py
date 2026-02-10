from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(scope="module")
def atoms():
    """
    A structure that includes hydrogen atoms.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1gya.bcif")
    return pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


@pytest.mark.benchmark
def benchmark_sasa(atoms):
    """
    Compute the SASA of each atom in a structure.
    """
    struc.sasa(atoms, vdw_radii="Single")
