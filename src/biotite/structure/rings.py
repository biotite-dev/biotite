# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions related to aromatic rings.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["find_aromatic_rings", "find_stacking_interactions", "PiStacking"]


from enum import IntEnum
import networkx as nx
import numpy as np
from biotite.structure.bonds import BondType
from biotite.structure.error import BadStructureError
from biotite.structure.geometry import displacement
from biotite.structure.util import norm_vector, vector_dot


class PiStacking(IntEnum):
    """
    The type of pi-stacking interaction.

    - ``PARALLEL``: parallel pi-stacking (also called *staggered* or *Sandwich*)
    - ``PERPENDICULAR``: perpendicular pi-stacking (also called *T-shaped*)
    """

    PARALLEL = 0
    PERPENDICULAR = 1


def find_aromatic_rings(atoms):
    """
    Find (anti-)aromatic rings in a structure.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to be searched for aromatic rings.
        Requires an associated :class:`BondList`.

    Returns
    -------
    rings : list of ndarray
        The indices of the atoms that form aromatic rings.
        Each ring is represented by a list of indices.
        Only rings with minimum size are returned, i.e. two connected rings
        (e.g. in tryptophan) are reported as separate rings.

    Notes
    -----
    This function does not distinguish between aromatic and antiaromatic rings.
    All cycles containing atoms that are completely connected by aromatic bonds
    are considered aromatic rings.

    The PDB *Chemical Component Dictionary* (CCD) does not identify aromatic rings in
    all compounds as such.
    Prominent examples are the nucleobases, where the 6-membered rings are not
    flagged as aromatic.

    Examples
    --------

    >>> nad = residue("NAD")
    >>> rings = find_aromatic_rings(nad)
    >>> print(rings)
    [array([41, 37, 36, 35, 43, 42]), array([19, 18, 16, 15, 21, 20]), array([12, 13, 14, 15, 21])]
    >>> for atom_indices in rings:
    ...     print(np.sort(nad.atom_name[atom_indices]))
    ['C2N' 'C3N' 'C4N' 'C5N' 'C6N' 'N1N']
    ['C2A' 'C4A' 'C5A' 'C6A' 'N1A' 'N3A']
    ['C4A' 'C5A' 'C8A' 'N7A' 'N9A']
    """
    if atoms.bonds is None:
        raise BadStructureError("Structure must have an associated BondList")
    bond_array = atoms.bonds.as_array()
    # To detect aromatic rings, only keep bonds that are aromatic
    aromatic_bond_array = bond_array[
        np.isin(
            bond_array[:, 2],
            [
                BondType.AROMATIC,
                BondType.AROMATIC_SINGLE,
                BondType.AROMATIC_DOUBLE,
                BondType.AROMATIC_TRIPLE,
            ],
        ),
        # We can omit the bond type now
        :2,
    ]
    aromatic_bond_graph = nx.from_edgelist(aromatic_bond_array.tolist())
    # Find the cycles with minimum size -> cycle basis
    rings = nx.cycle_basis(aromatic_bond_graph)
    return [np.array(ring, dtype=int) for ring in rings]


def find_stacking_interactions(
    atoms,
    centroid_cutoff=6.5,
    plane_angle_tol=np.deg2rad(30.0),
    shift_angle_tol=np.deg2rad(30.0),
):
    """
    Find pi-stacking interactions between aromatic rings.

    Parameters
    ----------
    atoms : AtomArray
        The atoms to be searched for aromatic rings.
        Requires an associated :class:`BondList`.
    centroid_cutoff : float
        The cutoff distance for ring centroids.
    plane_angle_tol : float
        The tolerance for the angle between ring planes that must be either
        parallel or perpendicular.
        Given in radians.
    shift_angle_tol : float
        The tolerance for the angle between the ring plane normals and the
        centroid difference vector.
        Given in radians.

    Returns
    -------
    interactions : list of tuple(ndarray, ndarray, PiStacking)
        The stacking interactions between aromatic rings.
        Each element in the list represents one stacking interaction.
        The first two elements of each tuple represent atom indices of the stacked
        rings.
        The third element of each tuple is the type of stacking interaction.

    See Also
    --------
    find_aromatic_rings : Used for finding the aromatic rings in this function.

    Notes
    -----
    This function does not distinguish between aromatic and antiaromatic rings.
    Furthermore, it does not distinguish between repulsive and attractive stacking:
    Usually, stacking two rings directly above each other is repulsive, as the pi
    orbitals above the rings repel each other, so a slight horizontal shift is
    usually required to make the interaction attractive.
    However, in details this is strongly dependent on heteroatoms and the exact
    orientation of the rings.
    Hence, this function aggregates all stacking interactions to simplify the
    conditions for pi-stacking.

    The conditions for pi-stacking are :footcite:`Wojcikowski2015` :

        - The ring centroids must be within cutoff `centroid_cutoff` distance.
          While :footcite:`Wojcikowski2015` uses a cutoff of 5.0 Å, 6.5 Å was
          adopted from :footcite:`Bouysset2021` to better identify perpendicular
          stacking interactions.
        - The planes must be parallel or perpendicular to each other within a default
          tolerance of 30°.
        - The angle between the plane normals and the centroid difference vector must be
          be either 0° or 90° within a default tolerance of 30°, to check for lateral
          shifts.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    Detect base stacking interactions in a DNA helix

    >>> from os.path import join
    >>> dna_helix = load_structure(
    ...     join(path_to_structures, "base_pairs", "1qxb.cif"), include_bonds=True
    ... )
    >>> interactions = find_stacking_interactions(dna_helix)
    >>> for ring_atom_indices_1, ring_atom_indices_2, stacking_type in interactions:
    ...     print(
    ...         dna_helix.res_id[ring_atom_indices_1[0]],
    ...         dna_helix.res_id[ring_atom_indices_2[0]],
    ...         PiStacking(stacking_type).name
    ...     )
    17 18 PARALLEL
    17 18 PARALLEL
    5 6 PARALLEL
    5 6 PARALLEL
    5 6 PARALLEL
    """
    rings = find_aromatic_rings(atoms)
    if len(rings) == 0:
        return []

    ring_centroids = np.array(
        [atoms.coord[atom_indices].mean(axis=0) for atom_indices in rings]
    )
    ring_normals = np.array(
        [_get_ring_normal(atoms.coord[atom_indices]) for atom_indices in rings]
    )

    # Create an index array that contains the Cartesian product of all rings
    indices = np.stack(
        [
            np.repeat(np.arange(len(rings)), len(rings)),
            np.tile(np.arange(len(rings)), len(rings)),
        ],
        axis=-1,
    )
    # Do not include duplicate pairs
    indices = indices[indices[:, 0] > indices[:, 1]]

    ## Condition 1: Ring centroids are close enough to each other
    diff = displacement(ring_centroids[indices[:, 0]], ring_centroids[indices[:, 1]])
    # Use squared distance to avoid time consuming sqrt computation
    sq_distance = vector_dot(diff, diff)
    is_interacting = sq_distance < centroid_cutoff**2
    indices = indices[is_interacting]

    ## Condition 2: Ring planes are parallel or perpendicular
    plane_angles = _minimum_angle(
        ring_normals[indices[:, 0]], ring_normals[indices[:, 1]]
    )
    is_parallel = _is_within_tolerance(plane_angles, 0, plane_angle_tol)
    is_perpendicular = _is_within_tolerance(plane_angles, np.pi / 2, plane_angle_tol)
    is_interacting = is_parallel | is_perpendicular
    indices = indices[is_interacting]
    # Keep in sync with the shape of the filtered indices,
    # i.e. after filtering, `is_parallel==False` means a perpendicular interaction
    is_parallel = is_parallel[is_interacting]

    ## Condition 3: The ring centroids are not shifted too much
    ## (in terms of normal-centroid angle)
    diff = displacement(ring_centroids[indices[:, 0]], ring_centroids[indices[:, 1]])
    norm_vector(diff)
    angles = np.stack(
        [_minimum_angle(ring_normals[indices[:, i]], diff) for i in range(2)]
    )
    is_interacting = (
        # For parallel stacking, the lateral shift may not exceed the tolerance
        (is_parallel & np.any(_is_within_tolerance(angles, 0, shift_angle_tol), axis=0))
        # For perpendicular stacking, one ring must be above the other,
        # but from the perspective of the other ring, the first ring is approximately
        # in the same plane
        | (
            ~is_parallel
            & (
                (
                    _is_within_tolerance(angles[0], 0, shift_angle_tol)
                    & _is_within_tolerance(angles[1], np.pi / 2, shift_angle_tol)
                )
                | (
                    _is_within_tolerance(angles[0], np.pi / 2, shift_angle_tol)
                    & _is_within_tolerance(angles[1], 0, shift_angle_tol)
                )
            )
        )
    )
    indices = indices[is_interacting]
    is_parallel = is_parallel[is_interacting]

    # Only return pairs of rings where all conditions were fulfilled
    return [
        (
            rings[ring_i],
            rings[ring_j],
            PiStacking.PARALLEL if is_parallel[i] else PiStacking.PERPENDICULAR,
        )
        for i, (ring_i, ring_j) in enumerate(indices)
    ]


def _get_ring_normal(ring_coord):
    """
    Get the normal vector perpendicular to the ring plane.

    Parameters
    ----------
    ring_coord : ndarray
        The coordinates of the atoms in the ring.

    Returns
    -------
    normal : ndarray
        The normal vector of the ring plane.
    """
    # Simply use any three atoms in the ring to calculate the normal vector
    # We can also safely assume that there are at least three atoms in the ring,
    # as otherwise it would not be a ring
    normal = np.cross(ring_coord[1] - ring_coord[0], ring_coord[2] - ring_coord[0])
    norm_vector(normal)
    return normal


def _minimum_angle(v1, v2):
    """
    Get the minimum angle between two vectors, i.e. the possible angle range is
    ``[0, pi/2]``.

    Parameters
    ----------
    v1, v2 : ndarray, shape=(n,3), dtype=float
        The vectors to measure the angle between.

    Returns
    -------
    angle : ndarray, shape=(n,), dtype=float
        The minimum angle between the two vectors.

    Notes
    -----
    This restriction is added here as the normal vectors of the ring planes
    have no 'preferred side'.
    """
    # Do not distinguish between the 'sides' of the rings -> take absolute of cosine
    return np.arccos(np.abs(vector_dot(v1, v2)))


def _is_within_tolerance(angles, expected_angle, tolerance):
    """
    Check if the angles are within a certain tolerance.

    Parameters
    ----------
    angles : ndarray, shape=x, dtype=float
        The angles to check.
    expected_angle : float
        The expected angle.
    tolerance : float
        The tolerance.

    Returns
    -------
    is_within_tolerance : ndarray, shape=x, dtype=bool
        True if the angles are within the tolerance, False otherwise.
    """
    return np.abs(angles - expected_angle) < tolerance
