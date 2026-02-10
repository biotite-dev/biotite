from pathlib import Path
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

PDB_ID = "1aki"


@pytest.fixture(autouse=True, scope="session")
def load_ccd():
    """
    Ensure that the CCD is already loaded to avoid biasing tests with its loading time.
    """
    info.get_ccd()


@pytest.fixture(scope="module")
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / f"{PDB_ID}.bcif")
    return pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


@pytest.fixture(scope="module")
def bond_array(atoms):
    return atoms.bonds.as_array()


@pytest.mark.benchmark
def benchmark_bond_list_creation(atoms, bond_array):
    """
    Create a `BondList` from an array of bonds, which involves sorting and deduplication.
    """
    struc.BondList(atoms.array_length(), bond_array)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "method",
    [
        struc.BondList.as_set,
        struc.BondList.as_graph,
        struc.BondList.as_array,
        struc.BondList.get_all_bonds,
        struc.BondList.adjacency_matrix,
        struc.BondList.bond_type_matrix,
    ],
    ids=lambda x: x.__name__,
)
def benchmark_conversion(atoms, method):
    """
    Convert the `BondList` to a different representation.
    """
    method(atoms.bonds)


@pytest.mark.benchmark
def benchmark_get_bonds(atoms):
    """
    Get the bonds for each atom index.
    """
    for i in range(atoms.array_length()):
        atoms.bonds.get_bonds(i)


@pytest.mark.benchmark
def benchmark_get_all_bonds(atoms):
    """
    Get the bonds for all atom indices.
    """
    atoms.bonds.get_all_bonds()


@pytest.mark.benchmark
def benchmark_concatenate(atoms):
    """
    Concatenate two `BondList` objects.
    """
    atoms.bonds.concatenate([atoms.bonds, atoms.bonds])


@pytest.mark.parametrize(
    "connect_fn", [struc.connect_via_distances, struc.connect_via_residue_names]
)
@pytest.mark.benchmark
def benchmark_connect(atoms, connect_fn):
    """
    Find bonds between atoms using the specified method.
    """
    connect_fn(atoms)


@pytest.mark.benchmark
def benchmark_find_connected(atoms):
    """
    Find all connected atoms for a given atom index.
    """
    struc.find_connected(atoms.bonds, 0)


@pytest.mark.benchmark
def benchmark_find_rotatable_bonds(atoms):
    """
    Find all rotatable bonds in a `BondList`.
    """
    struc.find_rotatable_bonds(atoms.bonds)
