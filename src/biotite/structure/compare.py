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

    Notes
    -----
    This function does not superimpose the subject to its reference.
    In most cases :func:`superimpose()` should be called prior to this
    function.

    Examples
    --------

    Calculate the RMSD of all models to the first model:

    >>> superimposed, _ = superimpose(atom_array, atom_array_stack)
    >>> print(rmsd(atom_array, superimposed))
    [2.912e-07 1.928e+00 2.103e+00 2.209e+00 1.806e+00 2.172e+00 2.704e+00
     1.360e+00 2.337e+00 1.818e+00 1.879e+00 2.471e+00 1.939e+00 2.035e+00
     2.167e+00 1.789e+00 1.653e+00 2.348e+00 2.247e+00 2.529e+00 1.583e+00
     2.115e+00 2.131e+00 2.050e+00 2.512e+00 2.666e+00 2.206e+00 2.397e+00
     2.328e+00 1.868e+00 2.316e+00 1.984e+00 2.124e+00 1.761e+00 2.642e+00
     1.721e+00 2.571e+00 2.579e+00]
    """
    return np.sqrt(np.mean(_sq_euclidian(reference, subject), axis=-1))


def rmsf(reference, subject):
    r"""
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

    Notes
    -----
    This function does not superimpose the subject to its reference.
    In most cases :func:`superimpose()` should be called prior to this
    function.

    Examples
    --------

    Calculate the :math:`C_\alpha` RMSF of all models to the average
    model:

    >>> ca = atom_array_stack[:, atom_array_stack.atom_name == "CA"]
    >>> ca_average = average(ca)
    >>> ca, _ = superimpose(ca_average, ca)
    >>> print(rmsf(ca_average, ca))
    [1.372 0.360 0.265 0.261 0.288 0.204 0.196 0.306 0.353 0.238 0.266 0.317
     0.358 0.448 0.586 0.369 0.332 0.396 0.410 0.968]
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
    coords = coord(atoms)
    if coords.ndim != 3:
        raise TypeError(
            "Expected an AtomArrayStack or an ndarray with shape (m,n,3)"
        )
    mean_coords = np.mean(coords, axis=0)
    if isinstance(atoms, AtomArrayStack):
        mean_array = atoms[0].copy()
        mean_array.coord = mean_coords
        return mean_array
    else:
        return mean_coords


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