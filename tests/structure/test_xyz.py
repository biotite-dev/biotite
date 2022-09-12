# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.
import itertools
import datetime
from tempfile import TemporaryFile
import glob
from os.path import join, split, splitext
import pytest
import numpy as np
import biotite.structure as struc
from biotite.structure import Atom, AtomArray, AtomArrayStack
import biotite.structure.info as info
import biotite.structure.io.xyz as xyz
from ..util import data_dir

# dictionary used for encoding prior knowledge regarding
# model count in different test cases
model_counts = {
    "10000_docked": 9,
    "10000_docked_1": 1,
    "10000_docked_2": 1,
    "ADP": 1,
    "aspirin_2d": 1,
    "aspirin_3d": 1,
    "BENZ": 1,
    "CO2": 6,
    "HArF": 1,
    "HWB": 1,
    "lorazepam": 1,
    "nu7026_conformers": 6,
    "TYR": 1,
    "CYN": 1,
    "zinc_33_conformers": 30,
}


def retrieve_file_name_from_path(path):
    """
    Like the name says this function is a helper to get
    the filename (without extension) from the absolute path.
    In order to access the model_counts.
    """
    if "/" in path:
        name = path.split("/")[-1]
    elif "\\" in path:
        name = path.split("\\")[-1]
    else:
        raise ValueError(
            "Neither '/' nor '\\' used as path seperators"
        )
        
    name = name.split(".")[0]
    return name


def draw_random_struct(N_atoms):
    """
    This function is used to draw a randoms AtomArray with given
    number of N_atoms

    Parameters
    ---
    N_atoms:
    """
    return struc.array(
        [
            Atom(
                [
                    float(np.random.rand()) for _ in range(3)
                ],
                element="C"
            ) for _ in range(N_atoms)
        ]
    )


def test_header_conversion():
    """
    Write known example data to the header of a xyz file and expect
    to retrieve the same information when reading the file again.
    """
    ref_names = (
        "Testxyz", "JD", "Biotite",
        str(datetime.datetime.now().replace(second=0, microsecond=0)),
        "3D", "Lorem", "Ipsum", "123", "Lorem ipsum dolor sit amet"
    )
    # draw this many random atoms for structure
    # necessary as header should only exist for a non empty XYZFile
    N_sample = 10
    for name in ref_names:
        xyz_file = xyz.XYZFile()
        xyz_file.set_structure(draw_random_struct(N_sample))
        xyz_file.set_header(name)
        temp = TemporaryFile("w+")
        xyz_file.write(temp)
        temp.seek(0)
        xyz_file = xyz.XYZFile.read(temp)
        atom_numbers, mol_names = xyz_file.get_header()
        temp.close()
        assert mol_names == name
        assert atom_numbers == N_sample


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
)
def test_model_count(path):
    """
    Test the get_model_count function based on known number of models
    from the test caes listed in the dictionary above.
    """
    name = retrieve_file_name_from_path(path)
    xyz_file = xyz.XYZFile.read(path)
    assert xyz.get_model_count(xyz_file) == model_counts[name]


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
)
def test_get_header(path):
    """
    Test the get_model_count function based on known number of models
    from the test caes listed in the dictionary above.
    """
    name = retrieve_file_name_from_path(path)
    xyz_file = xyz.XYZFile.read(path)

    if model_counts[name] > 1:
        atom_number, mol_name = xyz.get_header(xyz_file)
        assert type(atom_number) == list
        assert type(mol_name) == list
        assert len(atom_number) == len(mol_name)
        assert len(mol_name) == model_counts[name]
        # test if single header retrieve by model
        # yields same result
        for i in range(model_counts[name]):
            a_number_i, m_name_i = xyz.get_header(xyz_file, model=i)
            assert a_number_i == atom_number[i]
            assert m_name_i == mol_name[i]
    else:
        atom_number, mol_name = xyz.get_header(xyz_file)
        assert type(atom_number) == int
        assert type(mol_name) == str


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
)
def test_set_header(path):
    """
    Test if set header works on files with multiple models, based
    on model
    """
    name = retrieve_file_name_from_path(path)
    xyz_file = xyz.XYZFile.read(path)

    if model_counts[name] > 1:
        atom_number, mol_name = xyz.get_header(xyz_file)
        # test if single header retrieve by model
        # yields same result
        for i in range(model_counts[name]):
            xyz.set_header(xyz_file, str(i), model=i)
            atom_number_new, mol_name_new = xyz.get_header(xyz_file)
            assert atom_number_new[i] == atom_number[i]
            assert mol_name_new[i] == str(i)


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
)
def test_structure_conversion(path):
    """
    After reading a xyz file, writing the structure back to a new file
    and reading it again should give the same structure.

    In this case an SDF file is used, but it is compatible with the
    xyz format.
    """
    xyz_file = xyz.XYZFile.read(path)
    ref_atoms = xyz.get_structure(xyz_file)
    xyz_file = xyz.XYZFile()
    xyz.set_structure(xyz_file, ref_atoms)
    temp = TemporaryFile("w+")
    xyz_file.write(temp)
    # read previously written XYZFile
    temp.seek(0)
    xyz_file = xyz.XYZFile.read(temp)
    test_atoms = xyz.get_structure(xyz_file)
    temp.close()
    # test if correct instance
    instance_cond = isinstance(ref_atoms, AtomArray)
    instance_cond = instance_cond | isinstance(ref_atoms, AtomArrayStack)
    assert instance_cond
    # and if coordinates match
    assert test_atoms == ref_atoms


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "molecules", "*.xyz"))
)
def test_get_structure(path):
    """
    Test the get_structure function based on known number of models
    from the test cases listed in the dictionary above.
    """
    name = retrieve_file_name_from_path(path)
    xyz_file = xyz.XYZFile.read(path)
    if model_counts[name] > 1:
        struct = xyz.get_structure(xyz_file)
        assert isinstance(struct, AtomArrayStack)
        for i in range(model_counts[name]):
            struct_i = xyz.get_structure(xyz_file, model=i)
            assert type(struct_i) == AtomArray
            assert np.all(struct[i].coord == struct_i.coord)
    else:
        struct = xyz.get_structure(xyz_file)
        assert isinstance(struct, AtomArray)
