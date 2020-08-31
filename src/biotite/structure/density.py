# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions to calculate atomistic densities.
"""

__name__ = "biotite.structure"
__author__ = "Daniel Bauer"
__all__ = ["density"]

import numpy as np
from .atoms import coord


def density(atoms, selection=None, delta=1.0, bins=None,
            density=False, weights=None):
    r"""
    Compute the density of the selected atoms.

    This creates a 3d histogram over the coordinates of selected atoms.
    By default, the grid for the histogram is built based on the
    coordinates of the given `atoms` with an even gridspacing of
    `delta` in all three dimensions.
    Alternatively, a custom grid can be used.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The density is calculated based on these atoms.
        Alternatively, the coordinates can be directly provided as
        `ndarray`.
    selection : ndarray, dtype=bool, shape=(n,), optional
        Boolean mask for `atoms` to calculate the density only on a set
        of atoms.
    delta : float, optional
        Distance between grid points for density calculation (in Å).
    bins : int or sequence of scalars or str, optional
        Bins for the RDF.

        - If `bins` is an `int`, it defines the number of bins.
        - If `bins` is a sequence, it defines the bin edges, ignoring
          the actual coordinates of the `atoms` selection.
        - If `bins` is a string, it defines the function used to
          calculate the bins.

        See :func:`numpy.histogramdd()` for further details.
    density : boolean, optional
        If False, the number of samples in each bin is returned.
        Otherwise, returns the probability density function of each bin.
        See :func:`numpy.histogramdd()` for further details.
    weights: ndarray, shape=(n,) or shape=(m,n), optional
        An array of values to weight the contribution of *n* atoms in 
        *m* models.
        If the shape is *(n,)*, the weights will be interpreted as
        *per atom*.
        A shape of *(m,n)* allows to additionally weight atoms on a
        *per model* basis.
    
    Returns
    -------
    H : ndarray, dtype=float
        The threedimensional histogram of the selected atoms.
        The histogram takes the atoms in all models into account.
        The length of the histogram depends on `atoms` coordinates and
        `delta`, or the supplied `bins` input parameter.
    edges : list of ndarray, dtype=float
        A list containing the 3 arrays describing the bin edges.
    """
    coords = coord(atoms)
    
    is_stack = coords.ndim == 3

    # Define the grid for coordinate binning based on coordinates of
    # supplied atoms
    # This makes the binning independent of a supplied box vector and 
    # fluctuating box dimensions are not a problem
    # However, this means that the user has to make sure the region of
    # interest is in the center of the box, i.e. by centering the
    # investigated protein in the box.
    if bins is None:
        if is_stack:
            axis = (0, 1)
        else:
            axis = 0
        grid_min, grid_max = np.min(
            coords, axis=axis), np.max(coords, axis=axis
        )
        bins = [
            np.arange(grid_min[0], grid_max[0]+delta, delta),
            np.arange(grid_min[1], grid_max[1]+delta, delta),
            np.arange(grid_min[2], grid_max[2]+delta, delta),
        ]

    if selection is None:
        selected_coords = coords
    else:
        selected_coords = coords[...,selection, :]

    # Reshape the coords into Nx3
    coords = selected_coords.reshape((np.prod(selected_coords.shape[:-1]), 3))

    # We need a weight value per coordinate, but input might be per atom
    if weights is not None:
        if is_stack and len(weights.shape) < 2:
            weights = np.tile(weights, len(selected_coords))
        weights = weights.reshape(coords.shape[0])
    
    # Calculate the histogram
    hist = np.histogramdd(
        coords, bins=bins, density=density, weights=weights
    )
    return hist
