# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions related to working with the simulation box or unit cell
of a structure 
"""

__author__ = "Patrick Kunzmann"
__all__ = ["get_box_vectors", "get_box_volume", "repeat_box", "set_box_center",
           "coord_to_fraction", "fraction_to_coord"]

from numbers import Integral
import numpy as np
import numpy.linalg as linalg
from .util import vector_dot
from .atoms import AtomArray, AtomArrayStack


def get_box_vectors(len_a, len_b, len_c, alpha, beta, gamma):
    alpha = alpha * np.pi / 180
    beta  = beta  * np.pi / 180
    gamma = gamma * np.pi / 180

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
    # Using the triple product
    return np.abs(
        vector_dot(
            np.cross(
                box[..., 0, :], box[..., 1, :]
            ),
            box[..., 2, :]
        )
    )


def repeat_box(array, amount=1):
    if not isinstance(amount, Integral):
        raise TypeError("Amouint must be an integer")
    repeat_array = array.copy()
    for i in range(-amount, amount+1):
        for j in range(-amount, amount+1):
            for k in range(-amount, amount+1):
                # Omit the central box
                    if i != 0 or j != 0 or k != 0:
                        temp_array = array.copy()
                        # Shift coordinates to adjacent box/unit cell
                        translation_vec = np.sum(
                            array.box * np.array([i,j,k])[:, np.newaxis],
                            axis=-2
                        )
                        if isinstance(temp_array, AtomArray):
                            temp_array.coord += translation_vec
                        elif isinstance(temp_array, AtomArrayStack):
                            temp_array.coord += translation_vec[:,np.newaxis,:]
                        else:
                            raise TypeError(
                                "An atom array or stack is required"
                            )
                        repeat_array += temp_array
    return repeat_array, np.tile(
        np.arange(array.array_length),
        (1 + 2 * amount) ** 3
    )


def set_box_center(array, center):
    centered_array = array.copy()
    if len(center.shape) == 1:
        # Constant center for all frames
        centered_array.coord -= center
    if len(center.shape) == 2:
        if not isinstance(centered_array, AtomArrayStack):
            raise TypeError("An AtomArrayStack must be provided if multiple "
                            "centers are given")
        # Frame-wise centering
        centered_array.coord -= center[:, np.newaxis, :]
    # Put the center into the center of the box
    if isinstance(centered_array, AtomArray):
        centered_array.coord += np.sum(array.box/2, axis=-2)
    elif isinstance(centered_array, AtomArrayStack):
        centered_array.coord += np.sum(array.box/2, axis=-2)[:, np.newaxis, :]
    else:
        raise TypeError("An atom array or stack is required")
    # Move atoms outside the box into the box
    # (periodic boundary condition)
    fractions = coord_to_fraction(centered_array.coord, array.box)
    fractions_rem = fractions % 1
    centered_array.coord = fraction_to_coord(fractions_rem, array.box)
    return centered_array


def coord_to_fraction(coord, box):
    return np.matmul(coord, linalg.inv(box))


def fraction_to_coord(fraction, box):
    return np.matmul(fraction, box)