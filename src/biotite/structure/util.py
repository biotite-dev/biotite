# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Utility functions for in internal use in `Bio.Structure` package
"""

__author__ = "Patrick Kunzmann"
__all__ = ["vector_dot", "norm_vector", "distance"]

import numpy as np


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
        Scalar product with the dimension of the input arrays reduced by 1
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
    Calculate the distance between two position vectors
    
    Parameters
    ----------
    v1,v2 : ndarray
        The arrays to calculate the product from.
        The vectors are represented by the last axis.
    
    Returns
    -------
    product : float or ndarray
        vector distance with the dimension of the input arrays reduced by 1
    """
    dif = v1 - v2
    return np.sqrt((dif*dif).sum(axis=-1))