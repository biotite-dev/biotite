# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info


@pytest.fixture
def array():
    """
    Create an :class:`AtomArray` containing a lot of different
    molecules.
    The atoms that belong to a single molecule are not adjacent in the
    :class:`AtomArray`, but a are shuffled in random positions of the
    :class:`AtomArray`.
    """
    MOL_NAMES = [
        "ARG", # Molecule with multiple branches
        "TRP", # Molecule with a cycle
        "GLC", # Molecule with a cycle
        "NA",  # A single atom
        "ATP"  # Larger molecule
    ]
    N_MOLECULES = 20

    np.random.seed(0)
    
    atom_array = struc.AtomArray(0)
    for i, mol_name in enumerate(np.random.choice(MOL_NAMES, N_MOLECULES)):
        molecule = info.residue(mol_name)
        molecule.res_id[:] = i+1
        atom_array += molecule
    
    reordered_indices = np.random.choice(
        np.arange(atom_array.array_length()),
        atom_array.array_length(),
        replace=False
    )
    atom_array = atom_array[reordered_indices]

    return atom_array


@pytest.mark.parametrize(
    "as_stack, as_bonds",
    [
        (False, False),
        (True,  False),
        (False, True )
    ]
)
def test_get_molecule_indices(array, as_stack, as_bonds):
    """
    Multiple tests to :func:`get_molecule_indices()` on a
    :class:`AtomArray` of random molecules.
    """
    if as_stack:
        array = struc.stack([array])
    
    if as_bonds:
        test_indices = struc.get_molecule_indices(array.bonds)
    else:
        test_indices = struc.get_molecule_indices(array)
    
    seen_atoms = 0
    for indices in test_indices:
        molecule = array[..., indices]
        # Assert that all residue IDs in the molecule are equal
        # -> all atoms from the same molecule
        assert (molecule.res_id == molecule.res_id[0]).all()
        # Assert that no atom is missing from the molecule
        assert molecule.array_length() \
            == info.residue(molecule.res_name[0]).array_length()
        seen_atoms += molecule.array_length()
    # Assert that all molecules are fond
    assert seen_atoms == array.array_length()


@pytest.mark.parametrize(
    "as_stack, as_bonds",
    [
        (False, False),
        (True,  False),
        (False, True )
    ]
)
def test_get_molecule_masks(array, as_stack, as_bonds):
    """
    Test whether the masks returned by :func:`get_molecule_masks()`
    point to the same atoms as the indices returned by
    :func:`get_molecule_indices()`.
    """
    if as_stack:
        array = struc.stack([array])
    
    if as_bonds:
        ref_indices = struc.get_molecule_indices(array.bonds)
        test_masks = struc.get_molecule_masks(array.bonds)
    else:
        ref_indices = struc.get_molecule_indices(array)
        test_masks = struc.get_molecule_masks(array)
    
    for i in range(len(test_masks)):
        # Assert that the mask is 'True' for all indices
        # and that these 'True' values are the only ones in the mask
        assert (test_masks[i, ref_indices[i]] == True).all()
        assert np.count_nonzero(test_masks[i]) == len(ref_indices[i])


@pytest.mark.parametrize("as_stack", (False, True))
def test_molecule_iter(array, as_stack):
    """
    Test whether :func:`molecule_iter()` gives the same molecules as
    pointed by :func:`get_molecule_indices()`.
    """
    if as_stack:
        array = struc.stack([array])

    ref_indices = struc.get_molecule_indices(array)
    test_iterator = struc.molecule_iter(array)

    for i, molecule in enumerate(test_iterator):
        assert molecule == array[..., ref_indices[i]]