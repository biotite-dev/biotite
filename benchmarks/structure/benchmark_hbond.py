import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture(scope="module")
def atoms():
    """
    A structure that includes hydrogen atoms.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "1gya.bcif")
    return pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


@pytest.fixture(scope="module")
def atoms_without_bonds(atoms):
    """
    The same structure, but without a :class:`BondList`, so that bonded hydrogen
    atoms need to be identified via their distance to the donor.
    """
    atoms = atoms.copy()
    atoms.bonds = None
    return atoms


@pytest.mark.benchmark
def benchmark_hbond(atoms):
    """
    Find all hydrogen bonds in a structure using its bond information.
    """
    struc.hbond(atoms)


@pytest.mark.benchmark
def benchmark_hbond_without_bonds(atoms_without_bonds):
    """
    Find all hydrogen bonds in a structure that has no bond information.
    """
    with pytest.warns(UserWarning, match="no associated 'BondList'"):
        struc.hbond(atoms_without_bonds)
