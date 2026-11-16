import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.gro as gro
import biotite.structure.io.pdbx as pdbx
from biotite.structure.rdf import rdf
from tests.util import data_dir


@pytest.fixture(scope="module")
def atoms():
    """
    A small solvated structure.
    """
    gro_file = gro.GROFile.read(data_dir("structure") / "waterbox.gro")
    return gro_file.get_structure()[0]


@pytest.fixture(scope="module")
def large_atoms():
    """
    A large structure.
    """
    pdbx_file = pdbx.BinaryCIFFile.read(data_dir("structure") / "pdb" / "1gya.bcif")
    return pdbx.get_structure(pdbx_file, model=1)


@pytest.fixture(scope="module")
def random_atoms():
    """
    Random coordinates in a large box.
    """
    rng = np.random.default_rng(0)
    atoms = struc.AtomArray(30000)
    atoms.coord = rng.uniform(0, 80, size=(30000, 3))
    atoms.box = np.diag([80.0, 80.0, 80.0])
    return atoms


@pytest.mark.benchmark
def benchmark_rdf_few_centers(atoms):
    """
    Compute the RDF for a few centers in a small system.
    """
    rdf(atoms.coord[:1], atoms, interval=(0, 10), bins=100)


@pytest.mark.benchmark
def benchmark_rdf_many_centers(atoms):
    """
    Compute the RDF for many centers in a small system.
    """
    rdf(atoms.coord, atoms, interval=(0, 10), bins=100)


@pytest.mark.benchmark
def benchmark_rdf_large_system(large_atoms):
    """
    Compute the RDF for many centers in a large system.
    """
    rdf(large_atoms.coord, large_atoms, interval=(0, 10), bins=100)


@pytest.mark.benchmark
def benchmark_rdf_sparse_system(random_atoms):
    """
    Compute the RDF for few centers in a sparsely populated system.
    """
    rdf(random_atoms.coord[:10], random_atoms, interval=(0, 10), bins=100)
