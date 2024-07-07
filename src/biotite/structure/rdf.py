# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions to calculate the radial distribution function.
"""

__name__ = "biotite.structure"
__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["rdf"]

from numbers import Integral
import numpy as np
from biotite.structure.atoms import AtomArray, coord, stack
from biotite.structure.box import box_volume
from biotite.structure.celllist import CellList
from biotite.structure.geometry import displacement
from biotite.structure.util import vector_dot


def rdf(
    center, atoms, selection=None, interval=(0, 10), bins=100, box=None, periodic=False
):
    r"""
    Compute the radial distribution function *g(r)* (RDF) for one or
    multiple given central positions based on a given system of
    particles.

    Parameters
    ----------
    center : Atom or AtomArray or AtomArrayStack or ndarray, dtype=float
        Coordinates or atoms(s) to use as origin(s) for RDF calculation.

        - If a single :class:`Atom` or a :class:`ndarray` with shape
          *(3,)* is given, the RDF is only calculated for this position.
        - If an :class:`AtomArray` or a :class:`ndarray` with shape
          *(n,3)* is given, the calculated RDF histogram is an average
          over *n* postions.
        - If an :class:`AtomArrayStack` or a :class:`ndarray` with shape
          *(m,n,3)* is given, different centers are used for each model
          *m*.
          The calculated RDF histogram is an average over *m*
          models and *n* positions.
          This requires `atoms` to be an :class:`AtomArrayStack`.

    atoms : AtomArray or AtomArrayStack
        The distribution is calculated based on these atoms.
        When an an :class:`AtomArrayStack` is provided, the RDF
        histogram is averaged over all models.
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
        Must have shape *(3,3)* if atoms is an :class:`AtomArray` or
        *(m,3,3)* if atoms is an :class:`AtomArrayStack`, respectively.
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
    Calculate the oxygen-oxygen radial distribution function of water.
    The range of the histogram starts at 0.2 Å, in order to
    ignore the counts for the density for each oxygen to itself.

    >>> from os.path import join
    >>> waterbox = load_structure(join(path_to_structures, "waterbox.gro"))
    >>> oxygens = waterbox[:, waterbox.atom_name == 'OW']
    >>> bins, g_r = rdf(oxygens, oxygens, bins=49, interval=(0.2, 10), periodic=True)

    Print the RDF depending on the radius. Bins are in Å.

    >>> for x, y in zip(bins, g_r):
    ...     print(f"{x:.2f} {y:.2f}")
    0.30 0.00
    0.50 0.00
    0.70 0.04
    0.90 0.02
    1.10 0.03
    1.30 0.06
    1.50 0.03
    1.70 0.04
    1.90 0.04
    2.10 0.04
    2.30 0.04
    2.50 0.16
    2.70 1.99
    2.90 2.22
    3.10 1.34
    3.30 1.04
    3.50 0.97
    3.70 0.94
    3.90 0.97
    4.10 0.94
    4.30 0.98
    4.50 0.97
    4.70 0.96
    4.90 0.99
    5.10 0.99
    5.30 1.02
    5.50 1.02
    5.70 0.99
    5.90 0.98
    6.10 0.98
    6.30 0.99
    6.50 1.02
    6.70 1.02
    6.90 1.00
    7.10 1.01
    7.30 1.01
    7.50 1.00
    7.70 1.01
    7.90 0.99
    8.10 0.99
    8.30 0.99
    8.50 0.99
    8.70 0.99
    8.90 1.00
    9.10 1.01
    9.30 1.01
    9.50 1.00
    9.70 1.00
    9.90 0.99

    Find the radius for the first solvation shell.
    In this simple case, the density peak is identified by finding
    the maximum of the function.

    >>> peak_position = np.argmax(g_r)
    >>> print(f"{bins[peak_position]/10:.2f} nm")
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
    elif box.ndim == 2 and atoms.stack_depth() == 1:
        box = box[np.newaxis, :, :]

    center = coord(center)
    if center.ndim == 1:
        center = center.reshape((1, 1) + center.shape)
    elif center.ndim == 2:
        center = center.reshape((1,) + center.shape)

    if box.shape[0] != center.shape[0] or box.shape[0] != atom_coord.shape[0]:
        raise ValueError("Center, box, and atoms must have the same model count")

    # Calculate distance histogram
    edges = _calculate_edges(interval, bins)
    # Make histogram of quared distances to save computation time
    # of sqrt calculation
    sq_edges = edges**2
    threshold_dist = edges[-1]
    cell_size = threshold_dist
    disp = []
    for i in range(atoms.stack_depth()):
        # Use cell list to efficiently preselect atoms that are in range
        # of the desired bin range
        cell_list = CellList(atom_coord[i], cell_size, periodic, box[i])
        # 'cell_radius=1' is used in 'get_atoms_in_cells()'
        # This is enough to find all atoms that are in the given
        # interval (and more), since the size of each cell is as large
        # as the last edge of the bins
        near_atom_mask = cell_list.get_atoms_in_cells(center[i], as_mask=True)
        # Calculate distances of each center to preselected atoms
        # for each center
        for j in range(center.shape[1]):
            dist_box = box[i] if periodic else None
            # Calculate squared distances
            disp.append(
                displacement(
                    center[i, j], atom_coord[i, near_atom_mask[j]], box=dist_box
                )
            )
    # Make one array from multiple arrays with different length
    disp = np.concatenate(disp)
    sq_distances = vector_dot(disp, disp)
    hist, _ = np.histogram(sq_distances, bins=sq_edges)

    # Normalize with average particle density (N/V) in each bin
    bin_volume = (4 / 3 * np.pi * np.power(edges[1:], 3)) - (
        4 / 3 * np.pi * np.power(edges[:-1], 3)
    )
    n_frames = len(atoms)
    volume = box_volume(box).mean()
    density = atoms.array_length() / volume
    g_r = hist / (bin_volume * density * n_frames)

    # Normalize with number of centers
    g_r /= center.shape[1]

    bin_centers = (edges[:-1] + edges[1:]) * 0.5

    return bin_centers, g_r


def _calculate_edges(interval, bins):
    if isinstance(bins, Integral):
        if bins < 1:
            raise ValueError("At least one bin is required")
        return np.linspace(*interval, bins + 1)
    else:
        # 'bins' contains edges
        return np.array(bins, dtype=float)
