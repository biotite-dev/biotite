"""
This module provides functions to calculate the radial distribution function.
"""

__name__ = "biotite.structure"
__author__ = "Daniel Bauer, Patrick Kunzmann"
__all__ = ["rdf"]

from collections.abc import Sequence
from numbers import Integral
from typing import Any
import numpy as np
from biotite.rust.structure import CellList
from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
    Coord,
    MultiCoord,
    SingleCoord,
    coord,
    stack,
)
from biotite.structure.box import box_volume
from biotite.structure.geometry import displacement
from biotite.structure.util import vector_dot
from biotite.typing import XYZ, K, M, N, NDArray1, NDArray2, NDArray3

# Maximum number of coordinate elements in a single brute-force
# displacement evaluation, to bound the memory usage of the
# intermediate (n_centers, n_atoms, 3) array
_MAX_BRUTE_ELEMENTS = 30_000_000

# Creating a cell list and evaluating the distance to an atom from the
# cell list query result via fancy indexing is about this factor more
# expensive than the contiguous distance evaluation of the brute-force
# approach
_CELL_LIST_COST_FACTOR = 3


def rdf(
    center: SingleCoord | Coord[Any] | MultiCoord[Any, Any],
    atoms: AtomArray[N] | AtomArrayStack[M, N],
    selection: NDArray1[N, np.bool_] | None = None,
    interval: tuple[float, float] = (0, 10),
    bins: int | Sequence[float] | NDArray1[K, np.floating] = 100,
    box: NDArray2[XYZ, XYZ, np.floating]
    | NDArray3[M, XYZ, XYZ, np.floating]
    | None = None,
    periodic: bool = False,
    use_cell_list: bool | None = None,
) -> tuple[NDArray1[K, np.floating], NDArray1[K, np.floating]]:
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
          over *n* positions.
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
    bins : int or sequence of scalars, optional
        Bins for the RDF.

        - If `bins` is an `int`, it defines the number of bins for the
          given `interval`.
        - If `bins` is a sequence, it defines the bin edges, ignoring
          the `interval` parameter. The output `bins` has the length
          of this input parameter reduced by one.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
        Must have shape *(3,3)* if atoms is an :class:`AtomArray` or
        *(m,3,3)* if atoms is an :class:`AtomArrayStack`, respectively.
    periodic : bool, optional
        Defines if periodic boundary conditions are taken into account.
    use_cell_list : bool, optional
        Whether to use a :class:`CellList` for finding pairs of centers
        and atoms that may lie within the given `interval`.
        Creating a :class:`CellList` has a small overhead and its queries
        only pay off, if a sufficient number of pairwise distances is
        calculated.
        If this parameter is ``None`` (the default), the choice is made
        automatically based on the number of centers, the number of atoms
        and the bin range: The cell list is used, if the expected number
        of distance evaluations *n_centers \* n_atoms* is much larger
        than the number of atoms that actually need to be visited.

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
    >>> waterbox = load_structure(join(path_to_structure_data, "waterbox.gro"))
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
    # Promote `AtomArray` to a single-model stack so the rest of the
    # function can treat `atoms_stack` uniformly
    atoms_stack: AtomArrayStack[Any, Any] = (
        stack([atoms]) if isinstance(atoms, AtomArray) else atoms
    )
    if selection is not None:
        atoms_stack = atoms_stack[..., selection]

    atom_coord = atoms_stack.coord

    box_arr: NDArray3[M, XYZ, XYZ, np.floating]
    if box is None:
        if atoms_stack.box is None:
            raise ValueError("A box must be supplied")
        box_arr = atoms_stack.box  # pyright: ignore[reportAssignmentType]
    elif box.ndim == 2 and atoms_stack.stack_depth() == 1:
        box_arr = box[np.newaxis, :, :]
    else:
        box_arr = box  # pyright: ignore[reportAssignmentType]

    center_arr = coord(center)
    if center_arr.ndim == 1:
        center_arr = center_arr.reshape((1, 1) + center_arr.shape)
    elif center_arr.ndim == 2:
        center_arr = center_arr.reshape((1,) + center_arr.shape)

    if (
        box_arr.shape[0] != center_arr.shape[0]
        or box_arr.shape[0] != atom_coord.shape[0]
    ):
        raise ValueError("Center, box, and atoms must have the same model count")

    # Calculate distance histogram
    edges = _calculate_edges(interval, bins)
    # Make histogram of squared distances to save computation time
    # of sqrt calculation
    sq_edges = edges**2
    threshold_dist = edges[-1]
    cell_size = threshold_dist
    n_centers = center_arr.shape[1]  # pyright: ignore[reportGeneralTypeIssues]
    n_atoms = atom_coord.shape[1]

    if use_cell_list is None:
        if periodic:
            # The brute-force approach uses the minimum-image convention,
            # which can only find the closest periodic image of an atom
            # If the interval is larger than the box inradius, more
            # distant images may come within the interval, which only
            # the periodic cell list can find
            volumes = box_volume(box_arr)
            # The box vectors are the rows of the box matrices
            # The inradius of a parallelepiped is 3V/S,
            # where S is the total surface area
            face_areas = np.linalg.norm(
                np.cross(box_arr[:, [1, 2, 0]], box_arr[:, [2, 0, 1]], axis=-1),
                axis=-1,
            ).sum(axis=-1)
            inradius = (3 * volumes / face_areas).min()
            if threshold_dist >= inradius:
                use_cell_list = True
        if not use_cell_list:
            # A cell list query visits the atoms in the cell of the center
            # and the adjacent cells, i.e. roughly the atoms in a cube with
            # an edge length of three times the cell size
            swept_volume = (3 * threshold_dist) ** 3
            if periodic:
                visited_fraction = min(1.0, swept_volume / box_volume(box_arr).mean())
            else:
                # Without periodicity the coordinates are not bound to a
                # box, hence the extent of the system is estimated from
                # the bounding box of the coordinates
                coord_min = atom_coord.min(axis=1)
                coord_max = atom_coord.max(axis=1)
                bbox_volume = np.prod(
                    np.maximum(coord_max - coord_min, threshold_dist), axis=-1
                ).mean()
                visited_fraction = min(1.0, swept_volume / bbox_volume)
            # Creating the cell list requires one pass over all atoms and
            # the distances to the visited atoms are evaluated via fancy
            # indexing, which is more expensive than the contiguous
            # evaluation of the brute-force approach
            # Hence, the cell list pays off, if only a small fraction of
            # the atoms is visited per center and if there is a
            # sufficient number of centers to amortize the creation
            use_cell_list = (
                n_atoms
                + _CELL_LIST_COST_FACTOR * (n_centers * n_atoms * visited_fraction)
                < n_centers * n_atoms
            )

    disp = []
    for i in range(atoms_stack.stack_depth()):
        if use_cell_list:
            cell_list = CellList(atom_coord[i], cell_size, periodic, box_arr[i])
            # 'cell_radius=1' is used in 'get_atoms_in_cells()'
            # This is enough to find all atoms that are in the given
            # interval (and more), since the size of each cell is as large
            # as the last edge of the bins
            near_atom_pairs = cell_list.get_atoms_in_cells(
                center_arr[i], result_format=CellList.Result.PAIRS
            )
            dist_box = box_arr[i] if periodic else None
            # Each atom may appear multiple times in the query result,
            # if multiple periodic images of the atom lie in the
            # adjacent cells of a center
            # In this case the minimum-image convention would calculate
            # the same displacement for each image, leading to spurious
            # counts, hence the image duplicates are removed
            # Histogram counting is order-independent, hence the pairs
            # can be used in the order they are returned
            if periodic:
                # Periodic images of an atom may appear multiple times
                # in the query result with different atom indices
                # The minimum-image convention would calculate the same
                # displacement for each image, leading to spurious
                # counts, hence the duplicates are removed after
                # mapping the image indices back to the original atoms
                near_atom_pairs[:, 1] %= n_atoms
                unique_pairs = np.unique(near_atom_pairs, axis=0)
                disp.append(
                    displacement(
                        center_arr[i][unique_pairs[:, 0]],
                        atom_coord[i][unique_pairs[:, 1]],
                        box=dist_box,
                    )
                )
            else:
                disp.append(
                    displacement(
                        center_arr[i][near_atom_pairs[:, 0]],
                        atom_coord[i][near_atom_pairs[:, 1]],
                        box=dist_box,
                    )
                )
        else:
            dist_box = box_arr[i] if periodic else None
            max_centers = max(1, _MAX_BRUTE_ELEMENTS // n_atoms)
            for start in range(0, n_centers, max_centers):
                centers_chunk = center_arr[i, start : start + max_centers]
                disp.append(
                    displacement(
                        centers_chunk[:, np.newaxis, :],
                        atom_coord[i][np.newaxis, :, :],
                        box=dist_box,
                    ).reshape(-1, 3)
                )
    # Make one array from multiple arrays with different length
    disp_arr = np.concatenate(disp)
    sq_distances = vector_dot(disp_arr, disp_arr)
    hist, _ = np.histogram(sq_distances, bins=sq_edges)

    # Normalize with average particle density (N/V) in each bin
    bin_volume = (4 / 3 * np.pi * np.power(edges[1:], 3)) - (
        4 / 3 * np.pi * np.power(edges[:-1], 3)
    )
    n_frames = len(atoms_stack)
    volume = box_volume(box_arr).mean()
    density = atoms_stack.array_length() / volume
    g_r = hist / (bin_volume * density * n_frames)

    # Normalize with number of centers
    g_r /= center_arr.shape[1]  # pyright: ignore[reportGeneralTypeIssues]

    bin_centers = (edges[:-1] + edges[1:]) * 0.5

    return bin_centers, g_r


def _calculate_edges(
    interval: tuple[float, float],
    bins: int | Sequence[float] | NDArray1[Any, np.floating],
) -> NDArray1[Any, np.floating]:
    if isinstance(bins, Integral):
        if bins < 1:
            raise ValueError("At least one bin is required")
        return np.linspace(*interval, int(bins) + 1)
    else:
        # 'bins' contains edges
        return np.array(bins, dtype=float)
