# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
import sys
import warnings
from os.path import join, splitext
from tempfile import TemporaryFile
import numpy as np
import pytest
from pytest import approx
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdb.hybrid36 as hybrid36
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


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
        [False, True],
    ),
)
def test_array_conversion(path, model, hybrid36, include_bonds):
    pdb_file = pdb.PDBFile.read(path)
    # Test also the thin wrapper around the methods
    # 'get_structure()' and 'set_structure()'
    try:
        array1 = pdb.get_structure(pdb_file, model=model, include_bonds=include_bonds)
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
            match="Only positive integers can be converted into hybrid-36 notation",
        ):
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, array1, hybrid36=hybrid36)
        return
    else:
        pdb_file = pdb.PDBFile()
        pdb.set_structure(pdb_file, array1, hybrid36=hybrid36)

    array2 = pdb.get_structure(pdb_file, model=model, include_bonds=include_bonds)

    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert (
            array1.get_annotation(category).tolist()
            == array2.get_annotation(category).tolist()
        )
    assert array1.coord.tolist() == array2.coord.tolist()


@pytest.mark.parametrize(
    "path",
    glob.glob(join(data_dir("structure"), "*.pdb")),
)
def test_space_group(path):
    """
    Test the preservation of space group information and structure
    when reading and writing a PDB file.

    Parameters
    ----------
    path : str
        Path to the PDB file.
    """
    # Read the PDB file
    pdb_file = pdb.PDBFile.read(path)
    print(f"Testing file: {path}")

    try:
        # Extract structure and space group
        stack1 = pdb_file.get_structure(model=1)
        cryst1 = pdb_file.get_space_group()
    except biotite.InvalidFileError:
        raise

    # Write the structure and space group back to a new PDB file
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(stack1)
    pdb_file.set_space_group(cryst1)

    # Re-read the structure and space group
    stack2 = pdb_file.get_structure(model=1)
    cryst2 = pdb_file.get_space_group()

    # Assertions to check if the original and new data match
    assert stack1 == stack2, "Structure mismatch after writing and reading."
    assert cryst1 == cryst2, "Space group mismatch after writing and reading."


@pytest.mark.parametrize(
    "path, model",
    itertools.product(glob.glob(join(data_dir("structure"), "*.pdb")), [None, 1, -1]),
)
def test_pdbx_consistency(path, model):
    bcif_path = splitext(path)[0] + ".bcif"
    pdbx_file = pdbx.BinaryCIFFile.read(bcif_path)
    try:
        ref_atoms = pdbx.get_structure(pdbx_file, model=model, include_bonds=True)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    pdb_file = pdb.PDBFile.read(path)
    test_atoms = pdb_file.get_structure(model=model, include_bonds=True)

    # Some bond types are explicitly set only in the PDB file or the
    # PDBx file -> Remove bond types for comparison
    ref_atoms.bonds.remove_bond_order()
    test_atoms.bonds.remove_bond_order()

    if ref_atoms.box is not None:
        assert np.allclose(test_atoms.box, ref_atoms.box)
    try:
        assert test_atoms.bonds == ref_atoms.bonds
    except AssertionError:
        if model is None:
            ref_atoms = ref_atoms[0]
            test_atoms = test_atoms[0]
        test_bond_set = set([tuple(e) for e in test_atoms.bonds.as_array()])
        ref_bond_set = set([tuple(e) for e in ref_atoms.bonds.as_array()])
        print("Additional bonds in test_atoms:", file=sys.stderr)
        for i, j, _ in test_bond_set - ref_bond_set:
            print(test_atoms[i], file=sys.stderr)
            print(test_atoms[j], file=sys.stderr)
            print(file=sys.stderr)
        print("Additional bonds in ref_atoms:", file=sys.stderr)
        for i, j, _ in ref_bond_set - test_bond_set:
            print(ref_atoms[i], file=sys.stderr)
            print(ref_atoms[j], file=sys.stderr)
            print(file=sys.stderr)
        raise
    for category in ref_atoms.get_annotation_categories():
        assert (
            test_atoms.get_annotation(category).tolist()
            == ref_atoms.get_annotation(category).tolist()
        )
    assert test_atoms.coord.tolist() == ref_atoms.coord.tolist()


@pytest.mark.parametrize(
    "path, model",
    itertools.product(glob.glob(join(data_dir("structure"), "*.pdb")), [None, 1]),
)
def test_pdbx_consistency_assembly(path, model):
    """
    Check whether :func:`get_assembly()` gives the same result for the
    PDBx/mmCIF and PDB reader.
    """
    pdb_file = pdb.PDBFile.read(path)
    try:
        test_assembly = pdb.get_assembly(pdb_file, model=model)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    bcif_path = splitext(path)[0] + ".bcif"
    pdbx_file = pdbx.BinaryCIFFile.read(bcif_path)
    ref_assembly = pdbx.get_assembly(pdbx_file, model=model)

    for category in ref_assembly.get_annotation_categories():
        assert (
            test_assembly.get_annotation(category).tolist()
            == ref_assembly.get_annotation(category).tolist()
        )
    assert test_assembly.coord.flatten().tolist() == approx(
        ref_assembly.coord.flatten().tolist(), abs=1e-3
    )


@pytest.mark.parametrize("hybrid36", [False, True])
def test_extra_fields(hybrid36):
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)
    stack1 = pdb_file.get_structure(
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
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
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"]
    )

    assert stack1.ins_code.tolist() == stack2.ins_code.tolist()
    assert stack1.atom_id.tolist() == stack2.atom_id.tolist()
    assert stack1.b_factor.tolist() == approx(stack2.b_factor.tolist())
    assert stack1.occupancy.tolist() == approx(stack2.occupancy.tolist())
    assert stack1.charge.tolist() == stack2.charge.tolist()
    assert stack1 == stack2


@pytest.mark.filterwarnings("ignore")
def test_inferred_elements():
    # Read valid pdb file
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)
    stack = pdb_file.get_structure()

    # Remove all elements
    removed_stack = stack.copy()
    removed_stack.element[:] = ""

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
    itertools.product(glob.glob(join(data_dir("structure"), "*.pdb")), [None, 1, -1]),
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
    expected_box = np.array(
        [[[66.65, 0.00, 0.00], [0.00, 190.66, 0.00], [-24.59, 0.00, 68.84]]]
    )

    assert expected_box.flatten().tolist() == approx(a.box.flatten().tolist(), abs=1e-2)


def test_id_overflow():
    # Create an atom array >= 100k atoms
    length = 100000
    a = struc.AtomArray(length)
    a.coord = np.zeros(a.coord.shape)
    a.chain_id = np.full(length, "A")
    # Create residue IDs over 10000
    a.res_id = np.arange(1, length + 1)
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
    assert a2.array_length() == a.array_length()

    # Manually check if the written atom id is correct
    temp.seek(0)
    last_line = temp.readlines()[-1]
    atom_id = int(last_line.split()[1])
    assert atom_id == 1

    temp.close()

    # Write stack as hybrid-36 pdb file: no warning should be thrown
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        temp = TemporaryFile("w+")
        tmp_pdb_file = pdb.PDBFile()
        tmp_pdb_file.set_structure(a, hybrid36=True)
        tmp_pdb_file.write(temp)

    # Manually check if the output is written as correct hybrid-36
    temp.seek(0)
    last_line = temp.readlines()[-1]
    atom_id = last_line.split()[1]
    assert atom_id == "A0000"
    res_id = last_line.split()[4][1:]
    assert res_id == "BXG0"

    temp.close()


@pytest.mark.parametrize("model", [None, 1, 10])
def test_get_coord(model):
    # Choose a structure without inscodes and altlocs
    # to avoid atom filtering in reference atom array (stack)
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)

    ref_coord = pdb_file.get_structure(model=model).coord

    test_coord = pdb_file.get_coord(model=model)

    assert test_coord.shape == ref_coord.shape
    assert (test_coord == ref_coord).all()


@pytest.mark.parametrize("model", [None, 1, 10])
def test_get_b_factor(model):
    # Choose a structure without inscodes and altlocs
    # to avoid atom filtering in reference atom array (stack)
    path = join(data_dir("structure"), "1l2y.pdb")
    pdb_file = pdb.PDBFile.read(path)

    if model is None:
        # The B-factor is an annotation category
        # -> it can only be extracted in a per-model basis
        ref_b_factor = np.stack(
            [
                pdb_file.get_structure(model=m, extra_fields=["b_factor"]).b_factor
                for m in range(1, pdb_file.get_model_count() + 1)
            ]
        )
    else:
        ref_b_factor = pdb_file.get_structure(
            model=model, extra_fields=["b_factor"]
        ).b_factor

    test_b_factor = pdb_file.get_b_factor(model=model)

    assert test_b_factor.shape == ref_b_factor.shape
    assert (test_b_factor == ref_b_factor).all()


np.random.seed(0)
N = 200
LENGTHS = [3, 4, 5]


@pytest.mark.parametrize(
    "number, length",
    zip(
        list(
            itertools.chain(
                *[
                    np.random.randint(0, hybrid36.max_hybrid36_number(length), N)
                    for length in LENGTHS
                ]
            )
        ),
        list(itertools.chain(*[[length] * N for length in LENGTHS])),
    ),
)
def test_hybrid36_codec(number, length):
    """
    Test whether hybrid-36 encoding and subsequent decoding restores the
    same number.
    """
    string = hybrid36.encode_hybrid36(number, length)
    test_number = hybrid36.decode_hybrid36(string)
    assert test_number == number


def test_max_hybrid36_number():
    assert hybrid36.max_hybrid36_number(4) == 2436111
    assert hybrid36.max_hybrid36_number(5) == 87440031


@pytest.mark.parametrize("hybrid36", [False, True])
def test_bond_records(hybrid36):
    """
    Writing a structure with randomized bonds and reading them again
    should give a structure with the same bonds.
    """
    # Generate enough atoms to test the hybrid-36 encoding
    n_atoms = 200000 if hybrid36 else 10000
    atoms = struc.AtomArray(n_atoms)
    # NaN values cannot be written to PDB
    atoms.coord[...] = 0
    # Only the bonds of 'HETATM' atoms records are written
    atoms.hetero[:] = True
    # Omit time consuming element guessing
    atoms.element[:] = "NA"

    np.random.seed(0)
    # Create random bonds four times the number of atoms
    bond_array = np.random.randint(n_atoms, size=(4 * n_atoms, 2))
    # Remove bonds of atoms to themselves
    bond_array = bond_array[bond_array[:, 0] != bond_array[:, 1]]
    ref_bonds = struc.BondList(n_atoms, bond_array)
    atoms.bonds = ref_bonds

    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atoms, hybrid36)
    parsed_atoms = pdb_file.get_structure(model=1, include_bonds=True)
    test_bonds = parsed_atoms.bonds

    assert test_bonds == ref_bonds


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
    test_bonds.remove_bond_order()

    ref_bonds = struc.connect_via_residue_names(atoms)
    ref_bonds.remove_bond_order()

    assert test_bonds.as_set() == ref_bonds.as_set()


@pytest.mark.parametrize("model", [1, None])
def test_get_unit_cell(model):
    """
    Test :meth:`PDBFile.get_unit_cell()` on a known example with a simple
    space group and a single chain.
    """
    INVERSION_AXES = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    TRANSLATION_AXES = [(0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0)]

    path = join(data_dir("structure"), "1aki.pdb")
    pdb_file = pdb.PDBFile.read(path)
    original_structure = pdb_file.get_structure(model=model)
    if model is None:
        # The unit cell is the same for every model
        box = original_structure.box[0]
    else:
        box = original_structure.box
    cell_sizes = np.diagonal(box)

    unit_cell = pdb_file.get_unit_cell(model=model)

    # Space group has 4 copies in a unit cell
    assert unit_cell.array_length() == original_structure.array_length() * 4
    if model is None:
        assert unit_cell.stack_depth() == original_structure.stack_depth()
    for chain, inv_axes, trans_axes in zip(
        struc.chain_iter(unit_cell), INVERSION_AXES, TRANSLATION_AXES
    ):
        # Superimpose symmetry mates
        # by applying the appropriate transformations
        translation_vector = -0.5 * cell_sizes * trans_axes
        chain = struc.translate(chain, translation_vector)
        angles = np.array(inv_axes) * np.pi
        chain = struc.rotate(chain, angles)
        # Now both mates should be equal
        for category in original_structure.get_annotation_categories():
            assert (
                chain.get_annotation(category).tolist()
                == original_structure.get_annotation(category).tolist()
            )
        assert chain.coord.flatten().tolist() == approx(
            original_structure.coord.flatten().tolist(), abs=1e-3
        )


@pytest.mark.parametrize(
    "annotation, value, warning_only",
    [
        ("coord", -1000, False),
        ("coord", 10000, False),
        ("chain_id", "AA", False),
        ("res_id", 10000, True),
        ("res_name", "ABCD", False),
        ("atom_name", "ABCDE", False),
        ("atom_id", 100000, True),
        ("b_factor", -100, False),
        ("b_factor", 1000, False),
        ("occupancy", -100, False),
        ("occupancy", 1000, False),
        ("charge", -10, False),
        ("charge", 10, False),
    ],
)
def test_setting_incompatible_structure(annotation, value, warning_only):
    """
    Expect that an exception is raised, when trying to set a structure,
    with values exceeding the number of PDB columns
    """
    atoms = struc.AtomArray(10)
    atoms.coord[:, :] = 0.0
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = 1
    atoms.ins_code[:] = ""
    atoms.res_name[:] = "ABC"
    atoms.hetero[:] = False
    atoms.atom_name[:] = ["C{i}" for i in range(atoms.array_length())]
    atoms.element[:] = "C"
    atoms.set_annotation("atom_id", np.arange(atoms.array_length()) + 1)
    atoms.set_annotation("b_factor", np.full(atoms.array_length(), 0.0))
    atoms.set_annotation("occupancy", np.full(atoms.array_length(), 1.0))
    atoms.set_annotation("charge", np.full(atoms.array_length(), 0))

    # Set one annotation to a value that exceeds the number of columns
    if annotation == "coord":
        atoms.coord[0, 0] = value
    else:
        atoms.get_annotation(annotation)[0] = value

    pdb_file = pdb.PDBFile()
    if warning_only:
        with pytest.warns(UserWarning):
            pdb_file.set_structure(atoms)
    else:
        with pytest.raises(struc.BadStructureError):
            pdb_file.set_structure(atoms)
