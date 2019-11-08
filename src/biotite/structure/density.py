# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions to calculate atomistic densities.
"""

__author__ = "Daniel Bauer"
__all__ = ["density"]

from numbers import Integral
import numpy as np
from .atoms import Atom, AtomArray, stack, array, coord, AtomArrayStack
from .box import box_volume
from .geometry import displacement
from .util import vector_dot
from .celllist import CellList


def density(atoms, selection=None, delta=1.0, bins=None,
            density=False, weights=None):
    r"""
    Compute the density of an atoms selection.

    This creates a 3d histogram over the coordinates of selected atoms.
    By default, the grid for the histogram is built based on the
    coordinates of the given `atoms` with an even gridspacing of
    `delta` in all three
    dimensions.
    Alternatively, a custom grid can be used.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The density is calculated based on these atoms.
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

        See `numpy.histogramdd()` for further details.
    density : boolean, optional
        If False, the number of samples in each bin is returned.
        Otherwise, returns the probability density function of each bin.

        See `numpy.histogramdd()` for further details.
    weights: ndarray, shape=(N) or shape=(N,M), optional
        An array of values to weight the contribution of N atoms in 
        models.
        If the shape is (N), the weights will be interpreted as
        *per atom*. A shape of (N,M) allows to additionally weight atoms
        on a per model basis.
    
    Returns
    -------
    H : ndarray, dtype=float
        The threedimensional histogram of the selected atoms.
        The length of the histogram depends on `atoms` coordinates and
        `delta`, or the supplied `bins` input parameter.
    edges : list of ndarray, dtype=float
        A list containing the 3 arrays describing the bin edges.
    """
    is_stack = isinstance(atoms, AtomArrayStack)

    # Define the grid for coordinate binning based on coordinates of
    # supplied atoms
    # This makes the binning independent of a supplied box vector and 
    # fluctuating box dimensions are not a problem
    # However, this means that the user has to make sure the region of
    # interest is in the center of the box, i.e. by centering the to
    # investigated protein in the box.
    if bins is None:
        if is_stack:
            axis = (0, 1)
        else:
            axis = 0
        grid_min, grid_max = np.min(
            atoms.coord, axis=axis), np.max(atoms.coord, axis=axis
        )
        bins = np.array([
            np.arange(grid_min[0], grid_max[0]+delta, delta),
            np.arange(grid_min[1], grid_max[1]+delta, delta),
            np.arange(grid_min[2], grid_max[2]+delta, delta),
        ])

    if selection is None:
        selected = atoms
    else:
        selected = atoms[...,selection]

    # reshape the coords into Nx3
    coords = selected.coord.reshape((np.prod(selected.shape), 3))

    # We need a weight value per coordinate, but input might be per atom.
    if weights is not None:
        if is_stack and len(weights.shape) < 2:
            weights = np.repeat(weights, len(selected))
        weights = weights.reshape(coords.shape[0])
    
    # calculate the histogram
    hist = np.histogramdd(
        coords, bins=bins, density=density, weights=weights
    )
    return hist
