# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Utility functions for in internal use in `Bio.Structure` package
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["vector_dot", "norm_vector", "distance", "matrix_rotate"]

import numpy as np
from .atoms import Atom, array


def vector_dot(v1,v2):
    """
    Calculate vector dot product of two vectors.
    
    Parameters
    ----------
    v1,v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.
    
    Returns
    -------
    product : float or ndarray
        Scalar product over the last dimension of the arrays.
    """
    return (v1*v2).sum(axis=-1)


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
        

def distance(v1,v2):
    """
    Calculate the distance between two position vectors.
    
    Parameters
    ----------
    v1,v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.
    
    Returns
    -------
    product : float or ndarray
        Vector distance over the last dimension of the array.
    """
    dif = v1 - v2
    return np.sqrt((dif*dif).sum(axis=-1))


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

