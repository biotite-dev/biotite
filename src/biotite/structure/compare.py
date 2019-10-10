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
from .atoms import Atom, AtomArray, AtomArrayStack, coord
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
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        The reference structure.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        Structure(s) to be compared with `reference`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    
    Returns
    -------
    rmsd : float or ndarray, dtype=float, shape=(n,)
        RMSD between subject and reference.
        If subject is an :class:`AtomArray` a float is returned.
        If subject is an :class:`AtomArrayStack` an :class:`ndarray`
        containing the RMSD for each :class:`AtomArray` is returned.
    
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
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        The reference structure.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    subject : AtomArrayStack or ndarray, dtype=float, shape=(m,n,3)
        Structures to be compared with `reference`.
        The time *t* is represented by the models in the
        :class:`AtomArrayStack`.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    
    Returns
    -------
    rmsf : ndarray, dtype=float, shape=(n,)
        RMSF between subject and reference structure.
        The index corresponds to the atoms.
    
    See Also
    --------
    rmsd
    """
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=-2))


def average(atoms):
    """
    Calculate an average structure.
    
    The average structure has the average coordinates
    of the input models.
    
    Parameters
    ----------
    atoms : AtomArrayStack or ndarray, dtype=float, shape=(m,n,3)
        The structure models to be averaged.
        Alternatively, coordinates can be provided directly as
        :class:`ndarray`.
    
    Returns
    -------
    average : AtomArray or ndarray, dtype=float, shape=(n,3)
        Structure with averaged atom coordinates.
        If `atoms` is an :class:`ndarray` and :class:`ndarray` is also
        returned.
    
    See Also
    --------
    rmsd, rmsf
    
    Notes
    -----
    The calculated average structure is not suitable for visualization
    or geometric calculations, since bond lengths and angles will
    deviate from meaningful values.
    This method is rather useful to provide a reference structure for
    calculation of e.g. the RMSD or RMSF. 
    """
    coord = atoms.coord
    if coord.ndim != 3:
        raise TypeError(
            "Expected an AtomArrayStack or an ndarray with shape (m,n,3)"
        )
    mean_coord = np.mean(atoms.coord, axis=0)
    if isinstance(atoms, AtomArrayStack):
        mean_array = atoms[0].copy()
        mean_array.coord = mean_coord
        return mean_array
    else:
        return mean_coord


def _sq_euclidian(reference, subject):
    """
    Calculate squared euclidian distance between atoms in two
    structures.
    
    Parameters
    ----------
    reference : AtomArray or ndarray, dtype=float, shape=(n,3)
        Reference structure.
    subject : AtomArray or AtomArrayStack or ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        Structure(s) whose atoms squared euclidian distance to
        `reference` is measured.
    
    Returns
    -------
    ndarray, dtype=float, shape=(n,) or shape=(m,n)
        Squared euclidian distance between subject and reference.
        If subject is an :class:`AtomArray` a 1-D array is returned.
        If subject is an :class:`AtomArrayStack` a 2-D array is
        returned.
        In this case the first dimension indexes the AtomArray.
    """
    reference_coord = coord(reference)
    subject_coord = coord(subject)
    if reference_coord.ndim != 2:
        raise TypeError(
            "Expected an AtomArray or an ndarray with shape (n,3) as reference"
        )
    dif = subject_coord - reference_coord
    return vector_dot(dif, dif)