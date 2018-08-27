# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for calculation of characteristic values when
comparing multiple structures with each other.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["rmsd", "rmsf", "average"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack
from .util import vector_dot


def rmsd(reference, subject):
    """
    Calculate the RMSD between two structures.
    
    Calculate the root-mean-square-deviation (RMSD)
    of a structure compared to a reference structure.
    The RMSD is defined as:
    
    .. math:: RMSD = \\sqrt{ \\frac{1}{n} \\sum\\limits_{i=1}^n (x_i - x_{ref,i})^2}
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArray or AtomArrayStack
        Structure(s) to be compared with `reference`.
        `reference` and `subject` must have equal annotation arrays.
    
    Returns
    -------
    rmsd : float or ndarray, dtype=float, shape=(n,)
        RMSD between subject and reference.
        If subject is an `AtomArray` a float is returned.
        If subject is an `AtomArrayStack` an `ndarray`
        containing the RMSD for each `AtomArray` is returned.
    
    See Also
    --------
    rmsf
    """
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=-1))


def rmsf(reference, subject):
    """
    Calculate the RMSF between two structures.
    
    Calculate the root-mean-square-fluctuation (RMSF)
    of a structure compared to a reference structure.
    The RMSF is defined as:
    
    .. math:: RMSF(i) = \\sqrt{ \\frac{1}{T} \\sum\\limits_{t=1}^T (x_i(t) - x_{ref,i}(t))^2}
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArrayStack
        Structures to be compared with `reference`.
        reference` and `subject` must have equal annotation arrays.
        The time `t` is represented by the index of the first dimension
        of the AtomArrayStack.
    
    Returns
    -------
    rmsf : ndarray, dtype=float, shape=(n,)
        RMSF between subject and reference structure.
        The index corresponds to the atoms in the annotation arrays.
    
    See Also
    --------
    rmsd
    """
    if type(subject) != AtomArrayStack:
        raise ValueError("Subject must be AtomArrayStack")
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=0))


def average(atom_arrays):
    """
    Calculate an average structure.
    
    The average structure has the average coordinates
    of the input models.
    
    Parameters
    ----------
    atom_arrays : AtomArrayStack
        Stack of structures to be averaged
    
    Returns
    -------
    average : AtomArray
        Structure with averaged atom coordinates.
    
    See Also
    --------
    rmsd, rmsf
    
    Notes
    -----
    The calculated average structure is not suitable for visualisation
    or geometric calculations, since bond lengths and angles will
    deviate from meaningful values.
    This method is rather useful to provide a reference structure for
    calculation of e.g. the RMSD or RMSF. 
    """
    mean_array = atom_arrays[0].copy()
    mean_array.coord = np.mean(atom_arrays.coord, axis=0)
    return mean_array


def _sq_euclidian(reference, subject):
    """
    Calculate squared euclidian distance between atoms in two
    structures.
    
    Parameters
    ----------
    reference : AtomArray
        Reference structure.
    subject : AtomArray or AtomArrayStack
        Structure(s) whose atoms squared euclidian distance to
        `reference` is measured.
    
    Returns
    -------
    ndarray, dtype=float, shape=(n,) or shape=(m,n)
        Squared euclidian distance between subject and reference.
        If subject is an `AtomArray` a 1-D array is returned.
        If subject is an `AtomArrayStack` a 2-D array is returned.
        In this case the first dimension indexes the AtomArray.
    """
    if type(reference) != AtomArray:
        raise ValueError("Reference must be AtomArray")
    dif = subject.coord - reference.coord
    return vector_dot(dif, dif)