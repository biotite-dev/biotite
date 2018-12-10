# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions related to working with the simulation box or unit cell
of a structure 
"""

__author__ = "Patrick Kunzmann"
__all__ = ["get_box_vectors", "get_box_volume", "repeat_box", "center_in_box",
           "move_inside_box", "coord_to_fraction", "fraction_to_coord",
           "is_orthogonal"]

from numbers import Integral
import numpy as np
import numpy.linalg as linalg
from .util import vector_dot
from .atoms import AtomArray, AtomArrayStack


def get_box_vectors(len_a, len_b, len_c, alpha, beta, gamma):
    """
    Calculate the three vectors spanning a box from the unit cell
    lengths and angles.

    The return value of this function are the three box vectors as
    required for the `box` attribute in atom arrays and stacks.

    Parameters
    ----------
    len_a, len_b, len_c : float
        The lengths of the three box/unit cell vectors *a*, *b* and *c*.
    alpha, beta, gamma:
        The angles between the box vectors in radians.
        *alpha* is the angle between *b* and *c*,
        *beta* between *a* and *c*, *gamma* between *a* and *b*
    
    Returns
    -------
    box : ndarray, dtype=float, shape=(3,3)
    The three box vectors.
    The vector components are in the last dimension.
    The value can be directly used as `box` attribute in an atom array.
    """
    a_x = len_a
    b_x = len_b * np.cos(gamma)
    b_y = len_b * np.sin(gamma)
    c_x = len_c * np.cos(beta)
    c_y = len_c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    c_z = np.sqrt(len_c*len_c - c_x*c_x - c_y*c_y)
    box = np.array([
        [a_x,   0,   0],
        [b_x, b_y,   0],
        [c_x, c_y, c_z]
    ], dtype=float)
    
    # Fix numerical errors, as values, that are actually 0,
    # might not be calculated as such
    tol = 1e-6 * (len_a + len_b + len_c)
    box[np.abs(box) < tol] = 0
    
    return box


def get_box_volume(box):
    """
    Get the volume of one ore multiple boxes.

    Parameters
    ----------
    box : ndarray, shape=(3,3) or shape=(m,3,3)
        One or multiple boxes to get the volume for.
    
    Returns
    -------
    volume : float or ndarray, shape=(m,)
    """
    # Using the triple product
    return np.abs(
        vector_dot(
            np.cross(
                box[..., 0, :], box[..., 1, :]
            ),
            box[..., 2, :]
        )
    )


def repeat_box(atoms, amount=1):
    r"""
    Repeat the atoms in a box by duplicating and placing them in
    adjacent boxes.

    The output atom array (stack) contains the original atoms (central
    box) and duplicates of them in the given amount of adjacent boxes.
    The coordinates of the duplicate atoms are translated accordingly
    by the box coordinates.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to be repeated.
        If `atoms` is a `AtomArrayStack`, the atoms are repeated for
        each model, according to the box of each model.
    amount : int, optional
        The amount of boxes that are created in each direction of the
        central box.
        Hence, the total amount of boxes is
        :math:`(1 + 2 \cdot \text{amount}) ^ 3`.
        By default, one box is created in each direction, totalling in
        27 boxes.

    Returns
    -------
    repeated : AtomArray or AtomArrayStack
        The repeated atoms.
        Includes the original atoms (central box) in the beginning of
        the atom array (stack).
    indices : ndarray, dtype=int
        Indices to the atoms in the original atom array (stack).
    
    Examples
    --------

    >>> array = AtomArray(length=2)
    >>> array.coord = np.array([[1,5,3], [-1,2,5]], dtype=float)
    >>> array.box = np.array([[10,0,0], [0,10,0], [0,0,10]], dtype=float)
    >>> repeated, indices = repeat_box(array)
    >>> print(repeated.coord)
    [[  1.   5.   3.]
     [ -1.   2.   5.]
     [ -9.  -5.  -7.]
     [-11.  -8.  -5.]
     [ -9.  -5.   3.]
     [-11.  -8.   5.]
     [ -9.  -5.  13.]
     [-11.  -8.  15.]
     [ -9.   5.  -7.]
     [-11.   2.  -5.]
     [ -9.   5.   3.]
     [-11.   2.   5.]
     [ -9.   5.  13.]
     [-11.   2.  15.]
     [ -9.  15.  -7.]
     [-11.  12.  -5.]
     [ -9.  15.   3.]
     [-11.  12.   5.]
     [ -9.  15.  13.]
     [-11.  12.  15.]
     [  1.  -5.  -7.]
     [ -1.  -8.  -5.]
     [  1.  -5.   3.]
     [ -1.  -8.   5.]
     [  1.  -5.  13.]
     [ -1.  -8.  15.]
     [  1.   5.  -7.]
     [ -1.   2.  -5.]
     [  1.   5.  13.]
     [ -1.   2.  15.]
     [  1.  15.  -7.]
     [ -1.  12.  -5.]
     [  1.  15.   3.]
     [ -1.  12.   5.]
     [  1.  15.  13.]
     [ -1.  12.  15.]
     [ 11.  -5.  -7.]
     [  9.  -8.  -5.]
     [ 11.  -5.   3.]
     [  9.  -8.   5.]
     [ 11.  -5.  13.]
     [  9.  -8.  15.]
     [ 11.   5.  -7.]
     [  9.   2.  -5.]
     [ 11.   5.   3.]
     [  9.   2.   5.]
     [ 11.   5.  13.]
     [  9.   2.  15.]
     [ 11.  15.  -7.]
     [  9.  12.  -5.]
     [ 11.  15.   3.]
     [  9.  12.   5.]
     [ 11.  15.  13.]
     [  9.  12.  15.]]
    >>> print(indices)
    [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
     1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]
    """
    if not isinstance(amount, Integral):
        raise TypeError("The amount must be an integer")
    repeat_atoms = atoms.copy()
    for i in range(-amount, amount+1):
        for j in range(-amount, amount+1):
            for k in range(-amount, amount+1):
                # Omit the central box
                    if i != 0 or j != 0 or k != 0:
                        temp_atoms = atoms.copy()
                        # Shift coordinates to adjacent box/unit cell
                        translation_vec = np.sum(
                            atoms.box * np.array([i,j,k])[:, np.newaxis],
                            axis=-2
                        )
                        if isinstance(temp_atoms, AtomArray):
                            temp_atoms.coord += translation_vec
                        elif isinstance(temp_atoms, AtomArrayStack):
                            temp_atoms.coord += translation_vec[:,np.newaxis,:]
                        else:
                            raise TypeError(
                                "An atom array or stack is required"
                            )
                        repeat_atoms += temp_atoms
    return repeat_atoms, np.tile(
        np.arange(atoms.array_length()),
        (1 + 2 * amount) ** 3
    )


def center_in_box(atoms, center):
    centered_atoms = atoms.copy()
    if len(center.shape) == 1:
        # Constant center for all frames
        centered_atoms.coord -= center
    if len(center.shape) == 2:
        if not isinstance(centered_atoms, AtomArrayStack):
            raise TypeError("An AtomArrayStack must be provided if multiple "
                            "centers are given")
        # Frame-wise centering
        centered_atoms.coord -= center[:, np.newaxis, :]
    # Put the center into the center of the box
    if isinstance(centered_atoms, AtomArray):
        centered_atoms.coord += np.sum(atoms.box/2, axis=-2)
    elif isinstance(centered_atoms, AtomArrayStack):
        centered_atoms.coord += np.sum(atoms.box/2, axis=-2)[:, np.newaxis, :]
    else:
        raise TypeError("An atom array or stack is required")
    # Move atoms outside the box into the box
    # (periodic boundary condition)
    centered_atoms.coord = move_inside_box(
        centered_atoms.coord,
        centered_atoms.box
    )
    return centered_atoms


def move_inside_box(coord, box):
    fractions = coord_to_fraction(coord, box)
    fractions_rem = fractions % 1
    return fraction_to_coord(fractions_rem, box)


def coord_to_fraction(coord, box):
    return np.matmul(coord, linalg.inv(box))


def fraction_to_coord(fraction, box):
    return np.matmul(fraction, box)


def is_orthogonal(box):
    # Fix numerical errors, as values, that are actually 0,
    # might not be calculated as such
    tol = 1e-6
    return (np.abs(vector_dot(box[..., 0, :], box[..., 1, :])) < tol) & \
           (np.abs(vector_dot(box[..., 0, :], box[..., 2, :])) < tol) & \
           (np.abs(vector_dot(box[..., 1, :], box[..., 2, :])) < tol)