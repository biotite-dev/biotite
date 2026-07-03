import pytest
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

PDB_ID = "1aki"


@pytest.fixture(scope="module")
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(
        data_dir("structure") / "pdb" / f"{PDB_ID}.bcif"
    )
    return pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


@pytest.mark.benchmark
@pytest.mark.parametrize("method", [strucalph.to_3di, strucalph.to_protein_blocks])
def benchmark_structural_alphabet_methods(method, atoms):
    """
    Convert a structure to the given structural alphabet.
    """
    method(atoms)
