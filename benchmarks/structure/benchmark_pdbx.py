import itertools
from pathlib import Path
import pytest
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

PDB_ID = "1aki"


@pytest.fixture
def pdbx_file():
    return pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / f"{PDB_ID}.bcif")


@pytest.fixture
def deserialized_data(pdbx_file):
    categories = {}
    for category_name, category in pdbx_file.block.items():
        columns = {}
        for column_name, column in category.items():
            columns[column_name] = column.as_array()
        categories[category_name] = columns
    return categories


@pytest.fixture
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / f"{PDB_ID}.bcif")
    return pdbx.get_structure(pdbx_file, model=1, include_bonds=True)


@pytest.mark.benchmark
@pytest.mark.parametrize("format", ["cif", "bcif"])
def benchmark_deserialize_pdbx(format):
    """
    Deserialize all categories of a CIF or BinaryCIFFile.
    """
    path = Path(data_dir("structure")) / f"{PDB_ID}.{format}"
    if format == "cif":
        pdbx_file = pdbx.CIFFile.read(path)
    else:
        pdbx_file = pdbx.BinaryCIFFile.read(path)

    for _, category in pdbx_file.block.items():
        for _, column in category.items():
            column.as_array()


@pytest.mark.benchmark
@pytest.mark.parametrize("format", ["cif", "bcif"])
def benchmark_serialize_pdbx(deserialized_data, tmp_path, format):
    """
    Serialize all categories of a CIF or BinaryCIFFile.
    """
    if format == "cif":
        File = pdbx.CIFFile
        Block = pdbx.CIFBlock
        Category = pdbx.CIFCategory
    else:
        File = pdbx.BinaryCIFFile
        Block = pdbx.BinaryCIFBlock
        Category = pdbx.BinaryCIFCategory

    block = Block()
    for category_name, columns in deserialized_data.items():
        block[category_name] = Category(columns)

    pdbx_file = File()
    pdbx_file["structure"] = block
    pdbx_file.write(tmp_path / f"{PDB_ID}.{format}")


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "format, include_bonds", itertools.product(["cif", "bcif"], [False, True])
)
def benchmark_get_structure(format, include_bonds):
    """
    Parse a structure from a CIF or BinaryCIFFile.
    """
    path = Path(data_dir("structure")) / f"{PDB_ID}.{format}"
    if format == "cif":
        pdbx_file = pdbx.CIFFile.read(path)
    else:
        pdbx_file = pdbx.BinaryCIFFile.read(path)
    pdbx.get_structure(pdbx_file, model=1, include_bonds=include_bonds)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "format, include_bonds", itertools.product(["cif", "bcif"], [False, True])
)
def benchmark_set_structure(atoms, tmp_path, format, include_bonds):
    """
    Write a structure into a CIF or BinaryCIFFile.
    """
    if format == "cif":
        File = pdbx.CIFFile
    else:
        File = pdbx.BinaryCIFFile

    if not include_bonds:
        atoms.bonds = None

    pdbx_file = File()
    pdbx.set_structure(pdbx_file, atoms)
    pdbx_file.write(tmp_path / f"{PDB_ID}.{format}")


@pytest.mark.benchmark
def benchmark_compress(pdbx_file):
    """
    Compress a CIF file.
    """
    pdbx.compress(pdbx_file)
