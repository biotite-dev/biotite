import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(scope="module")
def arrays():
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "1aki.bcif")
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    return [atoms.copy() for _ in range(10)]


@pytest.mark.benchmark
@pytest.mark.parametrize("model_count", [2, 10])
def benchmark_stack(arrays, model_count):
    """Stack models with equal bond topology."""
    struc.stack(arrays[:model_count])
