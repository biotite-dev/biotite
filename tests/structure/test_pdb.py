# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from tempfile import TemporaryFile
import itertools
import glob
from os.path import join, splitext
import pytest
from pytest import approx
import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdb.hybrid36 as hybrid36
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io as io
from ..util import data_dir


def test_get_model_count():
    pdb_file = pdb.PDBFile.read(join(data_dir("structure"), "1l2y.pdb"))
    # Test also the thin wrapper around the method
    # 'get_model_count()'
    test_model_count = pdb.get_model_count(pdb_file)
    ref_model_count = pdb.get_structure(pdb_file).stack_depth()
    assert test_model_count == ref_model_count


@pytest.mark.parametrize(
    "path, model, hybrid36, include_bonds",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.pdb")),
        [None, 1, -1],
        [False, True],
        [False, True]
    )
)
def test_array_conversion(path, model, hybrid36, include_bonds):
    pdb_file = pdb.PDBFile.read(path)
    # Test also the thin wrapper around the methods
    # 'get_structure()' and 'set_structure()'
    try:
        array1 = pdb.get_structure(
            pdb_file, model=model, include_bonds=include_bonds
        )
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise
    
    if hybrid36 and (array1.res_id < 0).any():
        with pytest.raises(
            ValueError,
            match="Only positive integers can be converted "
                  "into hybrid-36 notation"
        ):
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, array1, hybrid36=hybrid36)
        return
    else:
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, array1, hybrid36=hybrid36)
    
    array2 = pdb.get_structure(
        pdb_file, model=model, include_bonds=include_bonds
    )
    
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()


@pytest.mark.parametrize(
    "path, model",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.pdb")),
        [None, 1, -1]
    )
)
def test_pdbx_consistency(path, model):
    cif_path = splitext(path)[0] + ".cif"
    pdb_file = pdb.PDBFile.read(path)
    try:
        a1 = pdb_file.get_structure(model=model)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    pdbx_file = pdbx.PDBxFile.read(cif_path)
    a2 = pdbx.get_structure(pdbx_file, model=model)
    
    if a2.box is not None:
        assert np.allclose(a1.box, a2.box)
    assert a1.bonds == a2.bonds
    for category in a1.get_annotation_categories():
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()
    assert a1.coord.tolist() == a2.coord.tolist()


@pytest.mark.parametrize("hybrid36", [False, True])
def test_extra_fields(hybrid36):
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)
    stack1 = pdb_file.get_structure(
        extra_fields=[
            "atom_id", "b_factor", "occupancy", "charge"
        ]
    )

    with pytest.raises(ValueError):
        pdb_file.get_structure(extra_fields=["unsupported_field"])
    
    # Add non-neutral charge values,
    # as the input PDB has only neutral charges
    stack1.charge[0] = -1
    stack1.charge[1] = 2

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(stack1, hybrid36=hybrid36)
    
    stack2 = pdb_file.get_structure(
        extra_fields=[
            "atom_id", "b_factor", "occupancy", "charge"
        ]
    )
    
    assert stack1.ins_code.tolist() == stack2.ins_code.tolist()
    assert stack1.atom_id.tolist() == stack2.atom_id.tolist()
    assert stack1.b_factor.tolist() == approx(stack2.b_factor.tolist())
    assert stack1.occupancy.tolist() == approx(stack2.occupancy.tolist())
    assert stack1.charge.tolist() == stack2.charge.tolist()
    assert stack1 == stack2


@pytest.mark.filterwarnings("ignore")
def test_guess_elements():
    # Read valid pdb file
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)
    stack = pdb_file.get_structure()

    # Remove all elements
    removed_stack = stack.copy()
    removed_stack.element[:] = ''

    # Save stack without elements to tmp file
    temp = TemporaryFile("w+")
    tmp_pdb_file = pdb.PDBFile()
    tmp_pdb_file.set_structure(removed_stack)
    tmp_pdb_file.write(temp)

    # Read new stack from file with guessed elements
    temp.seek(0)
    guessed_pdb_file = pdb.PDBFile.read(temp)
    temp.close()
    guessed_stack = guessed_pdb_file.get_structure()

    assert guessed_stack.element.tolist() == stack.element.tolist()


@pytest.mark.parametrize(
    "path, model",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.pdb")),
        [None, 1, -1]
    )
)
def test_box_shape(path, model):
    pdb_file = pdb.PDBFile.read(path)
    try:
        a = pdb_file.get_structure(model=model)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    if isinstance(a, struc.AtomArray):
        expected_box_dim = (3, 3)
    else:
        expected_box_dim = (len(a), 3, 3)
    assert expected_box_dim == a.box.shape


def test_box_parsing():
    path = join(data_dir("structure"), "1igy.pdb")
    pdb_file = pdb.PDBFile.read(path)
    a = pdb_file.get_structure()
    expected_box = np.array([[
        [66.65,   0.00, 0.00],
        [0.00,  190.66, 0.00],
        [-24.59,  0.00, 68.84]
    ]])

    assert expected_box.flatten().tolist() \
           == approx(a.box.flatten().tolist(), abs=1e-2)


def test_id_overflow():
    # Create an atom array >= 100k atoms
    length = 100000
    a = struc.AtomArray(length)
    a.coord = np.zeros(a.coord.shape)
    a.chain_id = np.full(length, "A")
    # Create residue IDs over 10000
    a.res_id = np.arange(1, length+1)
    a.res_name = np.full(length, "GLY")
    a.hetero = np.full(length, False)
    a.atom_name = np.full(length, "CA")
    a.element = np.full(length, "C")
    
    # Write stack to pdb file and make sure a warning is thrown
    with pytest.warns(UserWarning):
        temp = TemporaryFile("w+")
        pdb_file = pdb.PDBFile()
        pdb_file.set_structure(a)
        pdb_file.write(temp)

    # Assert file can be read properly
    temp.seek(0)
    a2 = pdb.get_structure(pdb.PDBFile.read(temp))
    assert(a2.array_length() == a.array_length())
    
    # Manually check if the written atom id is correct
    temp.seek(0)
    last_line = temp.readlines()[-1]
    atom_id = int(last_line.split()[1])
    assert(atom_id == 1)

    temp.close()
    
    # Write stack as hybrid-36 pdb file: no warning should be thrown
    with pytest.warns(None) as record:
        temp = TemporaryFile("w+")
        tmp_pdb_file = pdb.PDBFile()
        tmp_pdb_file.set_structure(a, hybrid36=True)
        tmp_pdb_file.write(temp)
    assert len(record) == 0

    # Manually check if the output is written as correct hybrid-36
    temp.seek(0)
    last_line = temp.readlines()[-1]
    atom_id = last_line.split()[1]
    assert(atom_id == "A0000")
    res_id = last_line.split()[4][1:]
    assert(res_id == "BXG0")

    temp.close()


@pytest.mark.parametrize("model", [None, 1, 10])
def test_get_coord(model):
    # Choose a structure without inscodes and altlocs
    # to avoid atom filtering in reference atom array (stack)
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)
    
    try:
        ref_coord = pdb_file.get_structure(model=model).coord
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise
    
    test_coord = pdb_file.get_coord(model=model)
    
    assert test_coord.shape == ref_coord.shape
    assert (test_coord == ref_coord).all()


np.random.seed(0)
N = 200
LENGTHS = [3, 4, 5]
@pytest.mark.parametrize(
    "number, length",
    zip(
        list(itertools.chain(*[
            np.random.randint(0, hybrid36.max_hybrid36_number(length), N)
            for length in LENGTHS
        ])),
        list(itertools.chain(*[
            [length] * N for length in LENGTHS
        ]))
    )
)
def test_hybrid36_codec(number, length):
    string = hybrid36.encode_hybrid36(number, length)
    test_number = hybrid36.decode_hybrid36(string)
    assert test_number == number


def test_max_hybrid36_number():
    assert hybrid36.max_hybrid36_number(4) == 2436111
    assert hybrid36.max_hybrid36_number(5) == 87440031


def test_bond_parsing():
    """
    Compare parsing of bonds from PDB with output from
    :func:`connect_via_residue_names()`.
    """
    # Choose a structure with CONECT records to test these as well
    path = join(data_dir("structure"), "3o5r.pdb")
    pdb_file = pdb.PDBFile.read(path)
    atoms = pdb.get_structure(pdb_file, model=1, include_bonds=True)
    
    test_bonds = atoms.bonds

    ref_bonds = struc.connect_via_residue_names(atoms)
    ref_bonds.remove_bond_order()

    assert test_bonds.as_set() == ref_bonds.as_set()