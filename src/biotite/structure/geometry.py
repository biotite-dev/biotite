# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for geometric measurements between atoms
in a structure, mainly lenghts and angles.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["displacement", "index_displacement", "distance", "index_distance",
           "angle", "index_angle", "dihedral", "index_dihedral",
           "dihedral_backbone", "centroid"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import vector_dot, norm_vector
from .filter import filter_backbone
from .box import (coord_to_fraction, fraction_to_coord,
                  move_inside_box, is_orthogonal)
from .error import BadStructureError


def displacement(atoms1, atoms2):
    """
    Measure the displacement vector, i.e. the vector difference, from
    one array of atom coordinates to another array of coordinates.
    
    Parameters
    ----------
    atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the displacement between.
        The vector from `atoms1` to `atoms2` is measured.
        The dimensions may vary.
        Alternatively an ndarray containing the coordinates can be
        provided.
        Usual *NumPy* broadcasting rules apply.
    
    Returns
    -------
    disp : ndarray
        The displacement vector(s). The shape is equal to the shape of
        the input `atoms` with the highest dimensionality minus the last
        axis.
    
    See also
    --------
    index_displacement
    """
    v1 = coord(atoms1)
    v2 = coord(atoms2)
    # Decide subtraction order based on shape, since an array can be
    # only subtracted by an array with less dimensions
    if len(v1.shape) <= len(v2.shape):
        return v2 - v1
    else:
        return -(v1 - v2)


def index_displacement(atoms, indices, periodic=False, box=None):
    """
    Measure the displacement, i.e. the vector difference, between pairs
    of atoms.

    The pairs refer to indices of a given atom array, whose pairwise
    displacement should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.
    In contrast to the `distance()` function, this function is able
    to take periodic boundary conditions into account.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The pairwise distances are calculated for these pairs.
        Alternatively, the atom coordinates can be directly provided as
        `ndarray`.
    indices : ndarray, shape=(k,2)
        Pairs of indices that point to `atoms`.
        The displacement is measured from ``indices[x,0]`` to 
        ``indices[x,1]``.
    periodic : bool, optional
        If set to true, periodic boundary conditions are taken into
        account (minimum-image convention).
        The `box` attribute of the `atoms` parameter is used for
        calculation.
        An alternative box can be provided via the `box` parameter.
        By default, periodicity is ignored.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    
    Returns
    -------
    disp : ndarray, shape=(k,) or shape=(m,k)
        The pairwise displacements.
        If `atoms` is an atom array stack, The distances are
        calculated for each model.
    
    Warnings
    --------
    In case `periodic` is set to true and if the box is not orthorhombic
    (at least one angle deviates from 90 degrees),
    the calculation requires approximately 8 times as long as in the
    orthorhombic case.
    Furthermore, it is not guaranteed, that the lowest-distance periodic
    copy is found for non-orthorhombic boxes; this is especially true
    for heavily skewed boxes.

    See also
    --------
    displacement
    """
    coord1, coord2 = _get_coord_from_indices(atoms, indices, 2)
    diff = coord2 - coord1
    
    if periodic:
        if box is None:
            if not isinstance(atoms, AtomArray) and \
               not isinstance(atoms, AtomArrayStack):
                    raise TypeError(
                        "An atom array or stack is required, if the box "
                        "parameter is not explicitly set"
                    )
            elif atoms.box is None:
                    raise ValueError(
                        "The atom array (stack) must have a box, if the box "
                        "parameter is not explicitly set"
                    )
            else:
                box = atoms.box
        # Transform difference vector
        # from coordinates into fractions of box vectors
        # for faster calculation laster on
        fractions = coord_to_fraction(diff, box)
        # Move vectors into box
        fractions = fractions % 1
        # Check for each model if the box vectors are orthogonal
        orthogonality = is_orthogonal(box)
        disp = np.zeros(fractions.shape, dtype=np.float64)
        if len(fractions.shape) == 2:
            # Single model
            if orthogonality:
                _displacement_orthogonal_box(
                    fractions, box, disp
                )
            else:
                _displacement_triclinic_box(
                    fractions.astype(np.float64, copy=False),
                    box.astype(np.float64, copy=False),
                    disp
                )
        elif len(fractions.shape) == 3:
            # Multiple models
            # (Model count) x (Atom count)
            for i in range(len(fractions)):
                if orthogonality[i]:
                    _displacement_orthogonal_box(
                        fractions[i], box[i], disp[i]
                    )
                else:
                    _displacement_triclinic_box(
                        fractions[i].astype(np.float64, copy=False),
                        box[i].astype(np.float64, copy=False),
                        disp[i]
                    )
        else:
            raise ValueError(
                f"{atoms.coord} is an invalid shape for atom coordinates"
            )
        return disp
    
    else:
        return diff



def distance(atoms1, atoms2):
    """
    Measure the euclidian distance between atoms.
    
    Parameters
    ----------
    atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the distances between.
        The dimensions may ary.
        Alternatively an ndarray containing the coordinates can be
        provided.
        Usual *NumPy* broadcasting rules apply.
    
    Returns
    -------
    dist : float or ndarray
        The atom distances.
        The shape is equal to the shape of the input `atoms` with the
        highest dimensionality minus the last axis.
    
    See also
    --------
    index_distance
    """
    diff = displacement(atoms1, atoms2)
    return np.sqrt(vector_dot(diff, diff))


def index_distance(atoms, indices, periodic=False, box=None):
    """
    Measure the euclidian distance between pairs of atoms.

    The pairs refer to indices of a given atom array, whose pairwise
    distances should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.
    In contrast to the `distance()` function, this function is able
    to take periodic boundary conditions into account.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The pairwise distances are calculated for these pairs.
        Alternatively, the atom coordinates can be directly provided as
        `ndarray`.
    indices : ndarray, shape=(k,2)
        Pairs of indices that point to `atoms`.
    periodic : bool, optional
        If set to true, periodic boundary conditions are taken into
        account (minimum-image convention).
        The `box` attribute of the `atoms` parameter is used for
        calculation.
        An alternative box can be provided via the `box` parameter.
        By default, periodicity is ignored.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    
    Returns
    -------
    dist : ndarray, shape=(k,) or shape=(m,k)
        The pairwise distances.
        If `atoms` is an atom array stack, The distances are
        calculated for each model.
    
    Warnings
    --------
    In case `periodic` is set to true and if the box is not orthorhombic
    (at least one angle deviates from 90 degrees),
    the calculation requires approximately 8 times as long as in the
    orthorhombic case.
    Furthermore, it is not guaranteed, that the lowest-distance periodic
    copy is found for non-orthorhombic boxes; this is especially true
    for heavily skewed boxes.

    See also
    --------
    distance
    """
    diff = index_displacement(atoms, indices, periodic, box)
    return np.sqrt(vector_dot(diff, diff))


def angle(atom1, atom2, atom3):
    """
    Measure the angle between 3 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the angle between. Alternatively an
        ndarray containing the coordinates can be provided.
    
    Returns
    -------
    angle : float or ndarray
        The angle(s) between the atoms. The shape is equal to the shape
        of the input `atoms` with the highest dimensionality minus the
        last axis.
    
    See also
    --------
    index_angle
    """
    v1 = coord(atom1) - coord(atom2)
    v2 = coord(atom3) - coord(atom2)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))


def index_angle(atoms, indices, periodic=False, box=None):
    """
    Measure the angle between triples of atoms.

    The triples refer to indices of a given atom array, whose triplewise
    angles should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The triplewise distances are calculated for these pairs.
        Alternatively, the atom coordinates can be directly provided as
        `ndarray`.
    indices : ndarray, shape=(k,3)
        Triples of indices that point to `atoms`.
    periodic : bool, optional
        If set to true, periodic boundary conditions are taken into
        account (minimum-image convention).
        The `box` attribute of the `atoms` parameter is used for
        calculation.
        An alternative box can be provided via the `box` parameter.
        By default, periodicity is ignored.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    
    Returns
    -------
    angle : ndarray, shape=(k,) or shape=(m,k)
        The triplewise angles.
        If `atoms` is an atom array stack, The distances are
        calculated for each model.
    
    Warnings
    --------
    In case `periodic` is set to true and if the box is not orthorhombic
    (at least one angle deviates from 90 degrees),
    the calculation requires approximately 8 times as long as in the
    orthorhombic case.
    Furthermore, it is not guaranteed, that the lowest-distance periodic
    copy is found for non-orthorhombic boxes; this is especially true
    for heavily skewed boxes.

    See also
    --------
    angle
    """
    v1 = index_displacement(atoms, indices[:, [1,0]], periodic, box)
    v2 = index_displacement(atoms, indices[:, [1,2]], periodic, box)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1,v2))


def dihedral(atom1, atom2, atom3, atom4):
    """
    Measure the dihedral angle between 4 atoms.
    
    Parameters
    ----------
    atoms1, atoms2, atoms3, atom4 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the dihedral angle between.
        Alternatively an ndarray containing the coordinates can be
        provided.
    
    Returns
    -------
    dihed : float or ndarray
        The dihedral angle(s) between the atoms. The shape is equal to
        the shape of the input `atoms` with the highest dimensionality
        minus the last axis.
    
    See Also
    --------
    index_dihedral
    dihedral_backbone
    """
    v1 = coord(atom2) - coord(atom1)
    v2 = coord(atom3) - coord(atom2)
    v3 = coord(atom4) - coord(atom3)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Calculation using atan2, to ensure the correct sign of the angle 
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    return np.arctan2(y,x)


def index_dihedral(atoms, indices, periodic=False, box=None):
    """
    Measure the dihedral angle between quadruples of atoms.

    The triples refer to indices of a given atom array, whose
    quadruplewise dihedral angles should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The quadruplewise dihedral angles are calculated for these
        pairs.
        Alternatively, the atom coordinates can be directly provided as
        `ndarray`.
    indices : ndarray, shape=(k,4)
        Quadruples of indices that point to `atoms`.
    periodic : bool, optional
        If set to true, periodic boundary conditions are taken into
        account (minimum-image convention).
        The `box` attribute of the `atoms` parameter is used for
        calculation.
        An alternative box can be provided via the `box` parameter.
        By default, periodicity is ignored.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, the given box is used instead of the
        `box` attribute of `atoms`.
    
    Returns
    -------
    dihedral : ndarray, shape=(k,) or shape=(m,k)
        The quadruplewise dihedral angles.
        If `atoms` is an atom array stack, The distances are
        calculated for each model.
    
    Warnings
    --------
    In case `periodic` is set to true and if the box is not orthorhombic
    (at least one angle deviates from 90 degrees),
    the calculation requires approximately 8 times as long as in the
    orthorhombic case.
    Furthermore, it is not guaranteed, that the lowest-distance periodic
    copy is found for non-orthorhombic boxes; this is especially true
    for heavily skewed boxes.

    See also
    --------
    dihedral
    dihedral_backbone
    """
    v1 = index_displacement(atoms, indices[:, [0,1]], periodic, box)
    v2 = index_displacement(atoms, indices[:, [1,2]], periodic, box)
    v3 = index_displacement(atoms, indices[:, [2,3]], periodic, box)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)
    
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Calculation using atan2, to ensure the correct sign of the angle 
    x = vector_dot(n1,n2)
    y = vector_dot(np.cross(n1,n2), v2)
    return np.arctan2(y,x)


def dihedral_backbone(atom_array, chain_id):
    """
    Measure the characteristic backbone dihedral angles of a structure.
    
    Parameters
    ----------
    atom_array: AtomArray or AtomArrayStack
        The protein structure. A complete backbone, without gaps,
        is required here.
    chain_id: string
        The ID of the polypeptide chain. The dihedral angles are
        calculated for ``atom_array[atom_array.chain_id == chain_id]``
    
    Returns
    -------
    phi, psi, omega : ndarray
        An array containing the 3 backbone dihedral angles for every
        CA. 'phi' is not defined at the N-terminus, 'psi' and 'omega'
        are not defined at the C-terminus. In these places the arrays
        have `NaN` values. If an `AtomArrayStack` is given, the output
        angles are 2-dimensional, the first dimension corresponds to
        the model number.
    
    Raises
    ------
    BadStructureError
        If the amount of backbone atoms is not equal to amount of
        residues times 3 (for N, CA and C).
    
    See Also
    --------
    dihedral
    
    Examples
    --------
    
    >>> phi, psi, omega = dihedral_backbone(atom_array, "A")
    >>> print(np.stack([phi * 360/(2*np.pi), psi * 360/(2*np.pi)]).T)
    [[          nan  -56.14491122]
     [ -43.98001079  -51.30875902]
     [ -66.46585868  -30.89801505]
     [ -65.21943089  -45.94467406]
     [ -64.74659263  -30.346291  ]
     [ -73.13553596  -43.42456851]
     [ -64.88203916  -43.25451315]
     [ -59.50867772  -25.69819463]
     [ -77.98930479   -8.82307681]
     [ 110.78405639    8.07924448]
     [  55.24420794 -124.37141223]
     [ -57.98304696  -28.76563093]
     [ -81.83404402   19.12508041]
     [-124.05653736   13.40120726]
     [  67.93147348   25.21773833]
     [-143.95159184  131.29701851]
     [ -70.10004605  160.06790798]
     [ -69.48368612  145.66883187]
     [ -77.26416822  124.22289316]
     [ -78.10009149           nan]]
    """
    # Filter all backbone atoms
    bb_coord = atom_array[...,
                            filter_backbone(atom_array) &
                            (atom_array.chain_id == chain_id)].coord
    if bb_coord.shape[-1] % 3 != 0:
        raise BadStructureError(
            "AtomArray has insufficient amount of backbone atoms "
            "(possibly missing terminus)"
        )
    
    # Coordinates for dihedral angle calculation
    # Dim 0: Model index (only for atom array stacks)
    # Dim 1: Angle index
    # Dim 2: X, Y, Z coordinates
    # Dim 3: Atoms involved in dihedral angle
    if isinstance(atom_array, AtomArray):
        angle_coord_shape = (len(bb_coord)//3, 3, 4)
    elif isinstance(atom_array, AtomArrayStack):
        angle_coord_shape = (bb_coord.shape[0], bb_coord.shape[1]//3, 3, 4)
    phi_coord   = np.full(angle_coord_shape, np.nan)
    psi_coord   = np.full(angle_coord_shape, np.nan)
    omega_coord = np.full(angle_coord_shape, np.nan)
    
    # Indices for coordinates of CA atoms 
    ca_i = np.arange(bb_coord.shape[-2]//3) * 3 + 1
    phi_coord  [..., 1: , :, 0]  = bb_coord[..., ca_i[1: ]-2 ,:]
    phi_coord  [..., 1: , :, 1]  = bb_coord[..., ca_i[1: ]-1 ,:]
    phi_coord  [..., 1: , :, 2]  = bb_coord[..., ca_i[1: ]   ,:]
    phi_coord  [..., 1: , :, 3]  = bb_coord[..., ca_i[1: ]+1 ,:]
    psi_coord  [..., :-1, :, 0]  = bb_coord[..., ca_i[:-1]-1 ,:]
    psi_coord  [..., :-1, :, 1]  = bb_coord[..., ca_i[:-1]   ,:]
    psi_coord  [..., :-1, :, 2]  = bb_coord[..., ca_i[:-1]+1 ,:]
    psi_coord  [..., :-1, :, 3]  = bb_coord[..., ca_i[:-1]+2 ,:]
    omega_coord[..., :-1, :, 0]  = bb_coord[..., ca_i[:-1]   ,:]
    omega_coord[..., :-1, :, 1]  = bb_coord[..., ca_i[:-1]+1 ,:]
    omega_coord[..., :-1, :, 2]  = bb_coord[..., ca_i[:-1]+2 ,:]
    omega_coord[..., :-1, :, 3]  = bb_coord[..., ca_i[:-1]+3 ,:]
    
    phi = dihedral(phi_coord[...,0], phi_coord[...,1],
                   phi_coord[...,2], phi_coord[...,3])
    psi = dihedral(psi_coord[...,0], psi_coord[...,1],
                   psi_coord[...,2], psi_coord[...,3])
    omega = dihedral(omega_coord[...,0], omega_coord[...,1],
                     omega_coord[...,2], omega_coord[...,3])
    
    return phi, psi, omega


def centroid(atoms):
    """
    Measure the centroid of a structure.
    
    Parameters
    ----------
    atoms: ndarray or AtomArray or AtomArrayStack
        The structures to determine the centroid from.
        Alternatively an ndarray containing the coordinates can be
        provided.
    
    Returns
    -------
    centroid : float or ndarray
        The centroid of the structure(s). `ndarray` is returned when
        an `AtomArrayStack` is given (centroid for each model).
    """
    return np.mean(coord(atoms), axis=-2)


def _get_coord_from_indices(atoms, indices, expected_amount):
    """
    Get the coordinates from an atom array (stack) at the given indices.
    Required for the `index_xxx()` methods.
    """
    if indices.shape[-1] != expected_amount:
        raise ValueError(
            f"Expected length {expected_amount} in the last dimension "
            f"of the indices, but got length {pairs.shape[-1]}"
        )
    coord_list = []
    for i in range(expected_amount):
        coord_list.append(coord(atoms)[..., indices[:,i], :])
    return tuple(coord_list)


def _displacement_orthogonal_box(fractions, box, disp):
    """
    Fill in the PBC-aware displacement vector for non-PBC-aware
    displacements given as fractions of given box vectors.
    """
    # Fraction components are guaranteed to be positive 
    # Use fraction vector components with lower absolute
    # -> new_vec[i] = vec[i] - 1 if vec[i] > 0.5 else vec[i]
    fractions[fractions > 0.5] -= 1
    disp[:] = fraction_to_coord(fractions, box)


def _displacement_triclinic_box(fractions, box, disp):
    """
    Fill in the PBC-aware displacement vector for non-PBC-aware
    displacements given as fractions of given box vectors.
    """
    diffs = fraction_to_coord(fractions, box)
    # Fraction components are guaranteed to be positive 
    # Test all 3 fraction vector components
    # with positive and negative sign
    # (i,j,k in {-1, 0})
    # Hence, 8 periodic copies are tested
    periodic_shift = []
    for i in range(-1, 1):
        for j in range(-1, 1):
            for k in range(-1, 1):
                x = i*box[0,0] + j*box[1,0] + k*box[2,0]
                y = i*box[0,1] + j*box[1,1] + k*box[2,1]
                z = i*box[0,2] + j*box[1,2] + k*box[2,2]
                periodic_shift.append([x,y,z])
    periodic_shift = np.array(periodic_shift, dtype=disp.dtype)
    # Create 8 periodically shifted variants for each atom
    shifted_diffs = diffs[:, np.newaxis, :] + periodic_shift[np.newaxis, :, :]
    # Find for each atom the periodically shifted variant with lowest
    # distance
    # Lowest squared distance -> lowest distance
    sq_distance = vector_dot(shifted_diffs, shifted_diffs)
    # for each given non-PBC-aware displacement find the PBC-aware
    # displacement with the lowest distance
    disp[:] = shifted_diffs[
        np.arange(len(shifted_diffs)),
        np.argmin(sq_distance, axis=1)
    ]