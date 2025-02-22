# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Utility functions for in internal use in `Bio.Structure` package.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "vector_dot",
    "norm_vector",
    "distance",
    "matrix_rotate",
    "coord_for_atom_name_per_residue",
]

import numpy as np
from biotite.structure.atoms import AtomArrayStack
from biotite.structure.error import BadStructureError
from biotite.structure.residues import get_residue_masks, get_residue_starts


def vector_dot(v1, v2):
    """
    Calculate vector dot product of two vectors.

    Parameters
    ----------
    v1, v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.

    Returns
    -------
    product : float or ndarray
        Scalar product over the last dimension of the arrays.
    """
    return (v1 * v2).sum(axis=-1)


def norm_vector(v):
    """
    Normalise a vector.

    Parameters
    ----------
    v : ndarray
        The array containg the vector(s).
        The vectors are represented by the last axis.
    """
    factor = np.linalg.norm(v, axis=-1)
    if isinstance(factor, np.ndarray):
        v /= factor[..., np.newaxis]
    else:
        v /= factor


def distance(v1, v2):
    """
    Calculate the distance between two position vectors.

    Parameters
    ----------
    v1, v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.

    Returns
    -------
    product : float or ndarray
        Vector distance over the last dimension of the array.
    """
    dif = v1 - v2
    return np.sqrt((dif * dif).sum(axis=-1))


def matrix_rotate(v, matrix):
    """
    Perform a rotation using a rotation matrix.

    Parameters
    ----------
    v : ndarray
        The coordinates to rotate.
    matrix : ndarray
        The rotation matrix.

    Returns
    -------
    rotated : ndarray
        The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


def coord_for_atom_name_per_residue(atoms, atom_names, mask=None):
    """
    Get the coordinates of a specific atom for every residue.

    If a residue does not contain the specified atom, the coordinates are `NaN`.
    If a residue contains multiple atoms with the specified name, an exception is
    raised.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The atom array or stack to get the residue-wise coordinates from.
    atom_names : list of str, length=k
        The atom names to get the coordinates for.
    mask : ndarray, shape=(n,), dtype=bool, optional
        A boolean mask to further select valid atoms from `atoms`.

    Returns
    -------
    coord: ndarray, shape=(k, m, r, 3) or shape=(k, r, 3)
        The coordinates of the specified atom for each residue.
    """
    is_multi_model = isinstance(atoms, AtomArrayStack)
    residue_starts = get_residue_starts(atoms)
    all_residue_masks = get_residue_masks(atoms, residue_starts)

    if is_multi_model:
        coord = np.full(
            (len(atom_names), atoms.stack_depth(), len(residue_starts), 3),
            np.nan,
            dtype=np.float32,
        )
    else:
        coord = np.full(
            (len(atom_names), len(residue_starts), 3),
            np.nan,
            dtype=np.float32,
        )

    for i, atom_name in enumerate(atom_names):
        specified_atom_mask = atoms.atom_name == atom_name
        if mask is not None:
            specified_atom_mask &= mask
        all_residue_masks_for_specified_atom = all_residue_masks & specified_atom_mask
        number_of_specified_atoms_per_residue = np.count_nonzero(
            all_residue_masks_for_specified_atom, axis=-1
        )
        if np.any(number_of_specified_atoms_per_residue > 1):
            raise BadStructureError(f"Multiple '{atom_name}' atoms per residue")
        residues_with_specified_atom = number_of_specified_atoms_per_residue == 1
        coord_of_specified_atoms = atoms.coord[..., specified_atom_mask, :]
        if is_multi_model:
            # Swap dimensions due to NumPy's behavior when using advanced indexing
            # (https://numpy.org/devdocs/user/basics.indexing.html#combining-advanced-and-basic-indexing)
            coord[i, ..., residues_with_specified_atom, :] = (
                coord_of_specified_atoms.transpose(1, 0, 2)
            )
        else:
            coord[i, residues_with_specified_atom, :] = coord_of_specified_atoms

    return coord
