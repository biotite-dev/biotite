from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(scope="module")
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1gya.bcif")
    return pdbx.get_structure(pdbx_file)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "method",
    [
        struc.superimpose,
        struc.superimpose_without_outliers,
        struc.superimpose_homologs,
        struc.superimpose_structural_homologs,
    ],
)
def benchmark_superimpose(method, atoms):
    """
    Compute superimposition of two structures with the same number of atoms.
    """
    method(atoms[0], atoms[1])
