# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions to calculate the radial distribution function.
"""

__author__ = "Daniel Bauer"
__all__ = ["rdf"]

from .atoms import Atom, AtomArray, stack, array, coord, AtomArrayStack
from .geometry import distance
from .box import box_volume
import numpy as np

def rdf(center, atoms, selection=None, interval=(0, 10), bins=100, box=None,
        periodic=False):
    r"""
    Compute the radial distribution function g(r) for a given point and a selection.

    Parameters
    ----------
    center : ndArray or Atom or AtomArray or AtomArrayStack
        Coordinates or Atoms(s) to use as origin(s) for rdf calculation
    atoms : AtomArray or AtomArrayStack
        Simulation cell to use for rdf calculation. Please not that atoms must
        have an associated box.
    selection : ndarray or None, optional
        Boolean mask for atoms to limit the RDF calculation on specific atoms
        (Default: None).
    interval : tuple or None, optional
        The range for the RDF in Angstroem (Default: (0, 10)).
    bins : int or sequence of scalars or str, optional
        Bins for the RDF. If bins is an int, it defines the number of bins for
        given range. If bins is a sequence, it defines the bin edges. If bins is
        a string, it defines the function used to calculate the bins
        (Default: 100).
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    periodic : bool, optional
        Defines if periodic boundary conditions are taken into account
        (Default: False).

    Returns
    -------
    bins : ndarray, dtype=float, shape=n
        The n bin coordinates of the RDF where n is defined by bins
    rdf : ndarry, dtype=float, shape=n
        RDF values for every bin

    Notes
    -----
    Since the RDF depends on the average particle density of the system, this
    function strictly requires an box.

    Examples
    --------
    TODO

    """
    if isinstance(atoms, AtomArray):
        # Reshape always to a stack for easier calculation
        atoms = stack([atoms])
    if selection is not None:
        atoms = atoms[..., selection]
    
    atom_coord = atoms.coord

    if box is None:
        if atoms.box is None:
            raise ValueError("A box must be supplied")
        else:
            box = atoms.box
    
    center = coord(center)
    if center.ndim == 1:
        center = center.reshape((1, 1) + center.shape)
    elif center.ndim == 2:
        center = center.reshape((1) + center.shape)
    
    if box.shape[0] != center.shape[0] or box.shape[0] != atom_coord.shape[0]:
        raise ValueError(
            "Center, box, and atoms must have the same model count"
        )

    # calculate distance histogram
    if periodic:
        distances = distance(center, atoms, box=box)
    else:
        distances = distance(center, atoms)
    hist, bin_edges = np.histogram(distances, range=interval, bins=bins)

    # Normalize with average particle density (N/V) in each bin
    bin_volume = (4 / 3 * np.pi * np.power(bin_edges[1:], 3)) - (4 / 3 * np.pi * np.power(bin_edges[:-1], 3))
    n_frames = len(atoms)
    volume = box_volume(box).mean()
    density = atoms.array_length() / volume
    g_r = hist / (bin_volume * density * n_frames)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    return bin_centers, g_r
