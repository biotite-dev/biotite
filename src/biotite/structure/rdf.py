# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions to calculate the radial distribution function.
"""

__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["rdf"]

from .atoms import Atom, AtomArray, stack, array, coord, AtomArrayStack
from .geometry import distance
from .box import box_volume
import numpy as np


def rdf(center, atoms, selection=None, interval=(0, 10), bins=100, box=None,
        periodic=False):
    r"""
    Compute the radial distribution function *g(r)* (RDF) for one or
    multiple given central positions based on a given system of
    particles.

    Parameters
    ----------
    center : Atom or AtomArray or AtomArrayStack or ndarray, dtype=float
        Coordinates or atoms(s) to use as origin(s) for RDF calculation.

        - If a single `Atom` or an `ndarray` with shape *(3,)* is given,
          the RDF is only calculated for this position.
        - If an `AtomArray` or an `ndarray` with shape *(n,3)* is given,
          the calculated RDF histogram is an average over *n*
          postions.
        - If an `AtomArrayStack` or an `ndarray` with shape *(m,n,3)* is
          given, different centers are used for each model *m*.
          The calculated RDF histogram is an average over *m*
          models and *n* positions.
          This requires `atoms` to be an `AtomArrayStack`.

    atoms : AtomArray or AtomArrayStack
        The distribution is calculated based on these atoms.
        When an an `AtomArrayStack` is provided, the RDF histogram is
        averaged over all models.
        Please not that `atoms` must have an associated box,
        unless `box` is set.
    selection : ndarray, dtype=bool, shape=(n,), optional
        Boolean mask for `atoms` to limit the RDF calculation to
        specific atoms.
    interval : tuple, optional
        The range in which the RDF is calculated.
    bins : int or sequence of scalars or str, optional
        Bins for the RDF.

        - If `bins` is an `int`, it defines the number of bins for the
          given `interval`.
        - If `bins` is a sequence, it defines the bin edges, ignoring
          the `interval` parameter. The output `bins` has the length
          of this input parameter reduced by one.
        - If `bins` is a string, it defines the function used to
          calculate the bins.

        See `numpy.histogram()` for further details.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
        Must have shape *(3,3)* if atoms is an `AtomArray` or
        *(m,3,3)* if atoms is an `AtomArrayStack`, respectively.
    periodic : bool, optional
        Defines if periodic boundary conditions are taken into account.

    Returns
    -------
    bins : ndarray, dtype=float, shape=n
        The centers of the histogram bins.
        The length of the array is given by the `bins` input parameter.
    rdf : ndarry, dtype=float, shape=n
        RDF values for every bin.

    Notes
    -----
    Since the RDF depends on the average particle density of the system,
    this function strictly requires an box.

    Examples
    --------
    Calculate the oxygen-oxygen radial distribution function of water:

    >>> s = io.load_structure("water.gro")
    >>> oxygens = s[:, s.atom_name == 'OW']
    >>> bins, g_r = rdf(oxygens, oxygens, bins=50, interval=(0, 10), \
    >>>     periodic=True)

    Print the RDF function. Print the RDF function.
    Bins are in Angstroem.

    >>> print("r    g_r")
    >>> for x, y in zip(bins, g_r):
    >>> print(f"{x:.2f} {y:.2f}")
    r    g_r
    0.10 889.50
    0.30 0.00
    0.50 0.00
    0.70 0.00
    0.90 0.00
    1.10 0.00
    1.30 0.00
    1.50 0.00
    1.70 0.00
    1.90 0.00
    2.10 0.00
    2.30 0.00
    2.50 0.10
    2.70 2.07
    2.90 2.31
    3.10 1.35
    3.30 1.03
    3.50 0.97
    3.70 0.94
    3.90 0.96
    4.10 0.96
    4.30 0.97
    4.50 0.96
    4.70 0.96
    4.90 0.99

    Find the radius for the first solvation shell
    
    >>> from scipy.signal import find_peaks
    >>> peak_positions = find_peaks(g_r)[0]
    >>> peak_positions = peak_positions[0]
    >>> print(f"{bins[peak_positions]/10:.2f} nm")
    0.29 nm
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

    # Calculate distance histogram
    dist_box = box if periodic else None
    if center.shape[1] > 1:
        distances = np.full((center.shape[1], atom_coord.shape[0],
                             atom_coord.shape[1]), np.nan)
        for c in range(center.shape[1]):
            distances[c] = distance(center[:, c:c+1, :],
                                    atom_coord,
                                    box=dist_box)
    else:
        distances = distance(center, atom_coord, box=dist_box)
    hist, bin_edges = np.histogram(distances, range=interval, bins=bins)

    # Normalize with average particle density (N/V) in each bin
    bin_volume =   (4 / 3 * np.pi * np.power(bin_edges[1: ], 3)) \
                 - (4 / 3 * np.pi * np.power(bin_edges[:-1], 3))
    n_frames = len(atoms)
    volume = box_volume(box).mean()
    density = atoms.array_length() / volume
    g_r = hist / (bin_volume * density * n_frames)

    # Normalize with number of centers
    g_r /= center.shape[1]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5

    return bin_centers, g_r
