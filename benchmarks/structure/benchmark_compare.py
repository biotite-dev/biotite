import itertools
from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1gya.bcif")
    atoms = pdbx.get_structure(pdbx_file)
    # Reduce the number of atoms to speed up the benchmark
    return atoms[..., atoms.element != "H"]


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "multi_model, aggregation",
    itertools.product([False, True], ["all", "chain", "residue", "atom"]),
)
def benchmark_lddt(atoms, multi_model, aggregation):
    """
    Compute lDDT on different aggregation levels.
    """
    reference = atoms[0]
    if multi_model:
        subject = atoms
    else:
        subject = atoms[0]

    struc.lddt(reference, subject, aggregation=aggregation)
