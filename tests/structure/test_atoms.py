# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pickle
import numpy as np
import pytest
import biotite.structure as struc


@pytest.fixture
def atom_list():
    chain_id = ["A", "A", "B", "B", "B"]
    res_id = [1, 1, 1, 1, 2]
    ins_code = [""] * 5
    res_name = ["ALA", "ALA", "PRO", "PRO", "MSE"]
    hetero = [False, False, False, False, True]
    atom_name = ["N", "CA", "O", "CA", "SE"]
    element = ["N", "C", "O", "C", "SE"]
    atom_list = []
    for i in range(5):
        atom_list.append(
            struc.Atom(
                [i, i, i],
                chain_id=chain_id[i],
                res_id=res_id[i],
                ins_code=ins_code[i],
                res_name=res_name[i],
                hetero=hetero[i],
                atom_name=atom_name[i],
                element=element[i],
            )
        )
    return atom_list


@pytest.fixture
def atom(atom_list):
    return atom_list[2]


@pytest.fixture
def array(atom_list):
    return struc.array(atom_list)


@pytest.fixture
def stack(array):
    return struc.stack([array, array.copy(), array.copy()])


@pytest.fixture
def array_box():
    return np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])


@pytest.fixture
def stack_box(stack, array_box):
    return np.array([array_box] * stack.stack_depth())


def test_shape(array, stack):
    assert array.shape == (5,)
    assert stack.shape == (3, 5)


def test_access(array):
    chain_id = ["A", "A", "B", "B", "B"]
    assert array.coord.shape == (5, 3)
    assert array.chain_id.tolist() == chain_id
    assert array.get_annotation("chain_id").tolist() == chain_id
    array.add_annotation("test1", dtype=int)
    assert array.test1.tolist() == [0, 0, 0, 0, 0]
    with pytest.raises(IndexError):
        array.set_annotation("test2", np.array([0, 1, 2, 3]))


def test_finding_compatible_dtype(array):
    """
    Check if a compatible dtype is selected, if the existing one is incompatible with
    the new annotation array.
    """
    # Has more than the default 4 characters
    CHAIN_ID = "LONG_ID"

    array.set_annotation("chain_id", np.array([CHAIN_ID] * array.array_length()))
    # Without a compatible dtype, the string would be truncated
    assert (array.chain_id[:] == CHAIN_ID).all()


def test_modification(atom, array, stack):
    new_atom = atom
    new_atom.chain_id = "C"
    del array[2]
    assert array.chain_id.tolist() == ["A", "A", "B", "B"]
    array[-1] = new_atom
    assert array.chain_id.tolist() == ["A", "A", "B", "C"]
    del stack[1]
    assert stack.stack_depth() == 2


def test_array_indexing(atom, array):
    filtered_array = array[array.chain_id == "B"]
    assert filtered_array.res_name.tolist() == ["PRO", "PRO", "MSE"]
    assert atom == filtered_array[0]
    filtered_array = array[[0, 2, 4]]
    assert filtered_array.element.tolist() == ["N", "O", "SE"]


def test_stack_indexing(stack):
    with pytest.raises(IndexError):
        stack[5]
    filtered_stack = stack[0]
    assert isinstance(filtered_stack, struc.AtomArray)
    filtered_stack = stack[0:2, stack.res_name == "PRO"]
    assert filtered_stack.atom_name.tolist() == ["O", "CA"]
    filtered_stack = stack[np.array([True, False, True])]
    assert filtered_stack.stack_depth() == 2
    assert filtered_stack.array_length() == 5
    filtered_stack = stack[:, 0]
    assert filtered_stack.stack_depth() == 3
    assert filtered_stack.array_length() == 1


def test_concatenate_single(array, stack):
    """
    Concatenation of a single array or stack should return the same object.
    """
    assert array == struc.concatenate([array])
    assert stack == struc.concatenate([stack])


def test_concatenation(array, stack):
    concat_array = array[2:] + array[:2]
    assert concat_array.chain_id.tolist() == ["B", "B", "B", "A", "A"]
    assert concat_array.coord.shape == (5, 3)
    concat_stack = stack[:, 2:] + stack[:, :2]
    assert concat_array.chain_id.tolist() == ["B", "B", "B", "A", "A"]
    assert concat_stack.coord.shape == (3, 5, 3)


def test_comparison(array):
    mod_array = array.copy()
    assert mod_array == array
    mod_array.coord += 1
    assert mod_array != array
    mod_array = array.copy()
    mod_array.res_name[0] = "UNK"
    mod_array != array


def test_bonds(array):
    assert array.bonds is None
    with pytest.raises(TypeError):
        # Expect a BondList
        array.bonds = 42
    with pytest.raises(ValueError):
        # Expect a BondList with array length as atom count
        array.bonds = struc.BondList(13)
    array.bonds = struc.BondList(5, np.array([(0, 1), (0, 2), (2, 3), (2, 4)]))
    assert array.bonds.as_array().tolist() == [
        [0, 1, 0],
        [0, 2, 0],
        [2, 3, 0],
        [2, 4, 0],
    ]
    filtered_array = array[array.chain_id == "B"]
    assert filtered_array.bonds.as_array().tolist() == [[0, 1, 0], [0, 2, 0]]
    concat_array = array + array
    assert concat_array.bonds.as_array().tolist() == [
        [0, 1, 0],
        [0, 2, 0],
        [2, 3, 0],
        [2, 4, 0],
        [5, 6, 0],
        [5, 7, 0],
        [7, 8, 0],
        [7, 9, 0],
    ]


def test_box(array, stack, array_box, stack_box):
    # Test attribute access
    with pytest.raises(ValueError):
        array.box = stack_box
    with pytest.raises(ValueError):
        array.box = np.array([42])
    with pytest.raises(ValueError):
        stack.box = np.array([42])
    array.box = array_box
    stack.box = stack_box
    # Test indexing
    assert (stack[0].box == array_box).all()
    assert (stack[:2].box == np.array([array_box] * 2)).all()
    assert (stack[:2, 3].box == np.array([array_box] * 2)).all()
    assert (stack[[True, False, True]].box == np.array([array_box] * 2)).all()


def test_array_from_atoms(atom_list):
    """
    Check whether custom annotations in :class:`Atom` objects are
    properly carried over to the :class:`AtomArray` when using
    :func:`array()`.
    """
    for atom in atom_list:
        atom.some_annotation = 42
    array = struc.array(atom_list)
    assert np.all(array.some_annotation == np.full(array.array_length(), 42))
    assert np.issubdtype(array.some_annotation.dtype, np.integer)


def test_pickle(atom, array, stack):
    """
    Check if pickling and unpickling works.
    This test is necessary since the classes implement the
    :meth:`__getattr__()` and :meth:`__setattr__()` methods.
    """
    test_atom = pickle.loads(pickle.dumps(atom))
    assert test_atom == atom

    test_array = pickle.loads(pickle.dumps(array))
    assert test_array == array

    test_stack = pickle.loads(pickle.dumps(stack))
    assert test_stack == stack


def test_set_print_limits(array, stack):
    """
    Check the output of :func:`set_print_limits()`
    by setting the maximum number of models and atoms to print very low.
    """
    atom_string = str(array[0])
    atom_repr = repr(array[0])
    struc.set_print_limits(max_models=1, max_atoms=1)
    assert str(array) == f"{atom_string}\n\t..."
    assert str(stack) == f"Model 1\n{atom_string}\n\t...\n\n...\n\n"
    assert repr(array) == f"array([\n\t{atom_repr},\n\t...,\n])"
    assert (
        repr(stack)
        == f"stack([\n\tarray([\n\t\t{atom_repr},\n\t\t...,\n\t]),\n\t...,\n])"
    )
