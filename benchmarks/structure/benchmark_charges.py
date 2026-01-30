import pytest
import biotite.structure as struc
import biotite.structure.info as info


@pytest.fixture(scope="module")
def atoms():
    return info.residue("PNN")


@pytest.mark.benchmark
def benchmark_partial_charges(atoms):
    """
    Compute the partial charges of each atom in a structure.
    """
    struc.partial_charges(atoms)
