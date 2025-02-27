# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for geometric measurements between atoms
in a structure, mainly lenghts and angles.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "displacement",
    "index_displacement",
    "distance",
    "index_distance",
    "angle",
    "index_angle",
    "dihedral",
    "index_dihedral",
    "dihedral_backbone",
    "centroid",
]

import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack, coord
from biotite.structure.box import coord_to_fraction, fraction_to_coord, is_orthogonal
from biotite.structure.filter import filter_amino_acids
from biotite.structure.util import (
    coord_for_atom_name_per_residue,
    norm_vector,
    vector_dot,
)


def displacement(atoms1, atoms2, box=None):
    """
    Measure the displacement vector, i.e. the vector difference, from
    one array of atom coordinates to another array of coordinates.

    Parameters
    ----------
    atoms1, atoms2 : ndarray, shape=(m,n,3) or ndarray, shape=(n,3) or ndarray, shape=(3,) or Atom or AtomArray or AtomArrayStack
        The atoms to measure the displacement between.
        The vector from `atoms1` to `atoms2` is measured.
        The dimensions may vary.
        Alternatively an ndarray containing the coordinates can be
        provided.
        Usual *NumPy* broadcasting rules apply.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        The shape *(m,3,3)* is only allowed, when the input coordinates
        comprise multiple models.

    Returns
    -------
    disp : ndarray, shape=(m,n,3) or ndarray, shape=(n,3) or ndarray, shape=(3,)
        The displacement vector(s). The shape is equal to the shape of
        the input `atoms` with the highest dimensionality.

    See Also
    --------
    index_displacement : The same calculation, but using atom indices.
    """
    v1 = coord(atoms1)
    v2 = coord(atoms2)
    # Decide subtraction order based on shape, since an array can be
    # only subtracted by an array with less dimensions
    if len(v1.shape) <= len(v2.shape):
        diff = v2 - v1
    else:
        diff = -(v1 - v2)

    # Use minimum-image convention if box is given
    if box is not None:
        # Transform difference vector
        # from coordinates into fractions of box vectors
        # for faster calculation laster on
        fractions = coord_to_fraction(diff, box)
        # Move vectors into box
        fractions = fractions % 1
        # Check for each model if the box vectors are orthogonal
        orthogonality = is_orthogonal(box)
        disp = np.zeros(fractions.shape, dtype=diff.dtype)
        if fractions.ndim == 1:
            # Single atom
            # Transform into two dimensions
            # to match signature of '_displacement_xxx()'
            fractions = fractions[np.newaxis, :]
            disp = disp[np.newaxis, :]
            if orthogonality:
                _displacement_orthogonal_box(fractions, box, disp)
            else:
                _displacement_triclinic_box(
                    fractions.astype(diff.dtype, copy=False),
                    box.astype(diff.dtype, copy=False),
                    disp,
                )
            # Transform back
            disp = disp[0]
        if fractions.ndim == 2:
            # Single model
            if orthogonality:
                _displacement_orthogonal_box(fractions, box, disp)
            else:
                _displacement_triclinic_box(
                    fractions.astype(diff.dtype, copy=False),
                    box.astype(diff.dtype, copy=False),
                    disp,
                )
        elif fractions.ndim == 3:
            # Multiple models
            # (Model count) x (Atom count)
            for i in range(len(fractions)):
                if box.ndim == 2:
                    box_for_model = box
                    orthogonality_for_model = orthogonality
                elif box.ndim == 3:
                    box_for_model = box[i]
                    orthogonality_for_model = orthogonality[i]
                else:
                    raise ValueError(f"{box.ndim} are to many box dimensions")
                if orthogonality_for_model:
                    _displacement_orthogonal_box(fractions[i], box_for_model, disp[i])
                else:
                    _displacement_triclinic_box(
                        fractions[i].astype(diff.dtype, copy=False),
                        box_for_model.astype(diff.dtype, copy=False),
                        disp[i],
                    )
        else:
            raise ValueError(f"{diff.shape} is an invalid shape for atom coordinates")
        return disp

    else:
        return diff


def index_displacement(*args, **kwargs):
    """
    index_displacement(atoms, indices, periodic=False, box=None)

    Measure the displacement, i.e. the vector difference, between pairs
    of atoms.

    The pairs refer to indices of a given atom array, whose pairwise
    displacement should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.
    In contrast to the :func:`distance()` function, this function is
    able to take periodic boundary conditions into account.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The pairwise distances are calculated for these pairs.
        Alternatively, the atom coordinates can be directly provided as
        :class:`ndarray`.
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

    See Also
    --------
    displacement
    """
    return _call_non_index_function(displacement, 2, *args, **kwargs)


def distance(atoms1, atoms2, box=None):
    """
    Measure the euclidian distance between atoms.

    Parameters
    ----------
    atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the distances between.
        The dimensions may vary.
        Alternatively an ndarray containing the coordinates can be
        provided.
        Usual *NumPy* broadcasting rules apply.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        The shape *(m,3,3)* is only allowed, when the input coordinates
        comprise multiple models.

    Returns
    -------
    dist : float or ndarray
        The atom distances.
        The shape is equal to the shape of the input `atoms` with the
        highest dimensionality minus the last axis.

    See Also
    --------
    index_distance : The same calculation, but using atom indices.
    """
    diff = displacement(atoms1, atoms2, box)
    return np.sqrt(vector_dot(diff, diff))


def index_distance(*args, **kwargs):
    """
    index_distance(atoms, indices, periodic=False, box=None)

    Measure the euclidian distance between pairs of atoms.

    The pairs refer to indices of a given atom array, whose pairwise
    distances should be calculated.
    If an atom array stack is provided, the distances are calculated for
    each frame/model.
    In contrast to the :func:`distance()` function, this function is
    able to take periodic boundary conditions into account.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or ndarray, shape=(n,3) or shape=(m,n,3)
        The atoms the `indices` parameter refers to.
        The pairwise distances are calculated for these pairs.
        Alternatively, the atom coordinates can be directly provided as
        :class:`ndarray`.
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

    See Also
    --------
    distance
    """
    return _call_non_index_function(distance, 2, *args, **kwargs)


def angle(atoms1, atoms2, atoms3, box=None):
    """
    Measure the angle between 3 atoms.

    Parameters
    ----------
    atoms1, atoms2, atoms3 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the angle between. Alternatively an
        ndarray containing the coordinates can be provided.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        The shape *(m,3,3)* is only allowed, when the input coordinates
        comprise multiple models.

    Returns
    -------
    angle : float or ndarray
        The angle(s) between the atoms. The shape is equal to the shape
        of the input `atoms` with the highest dimensionality minus the
        last axis.

    See Also
    --------
    index_angle : The same calculation, but using atom indices.
    """
    v1 = displacement(atoms1, atoms2, box)
    v2 = displacement(atoms3, atoms2, box)
    norm_vector(v1)
    norm_vector(v2)
    return np.arccos(vector_dot(v1, v2))


def index_angle(*args, **kwargs):
    """
    index_angle(atoms, indices, periodic=False, box=None)

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
        :class:`ndarray`.
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

    See Also
    --------
    angle
    """
    return _call_non_index_function(angle, 3, *args, **kwargs)


def dihedral(atoms1, atoms2, atoms3, atoms4, box=None):
    """
    Measure the dihedral angle between 4 atoms.

    Parameters
    ----------
    atoms1, atoms2, atoms3, atoms4 : ndarray or Atom or AtomArray or AtomArrayStack
        The atoms to measure the dihedral angle between.
        Alternatively an ndarray containing the coordinates can be
        provided.
    box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        If this parameter is set, periodic boundary conditions are
        taken into account (minimum-image convention), based on
        the box vectors given with this parameter.
        The shape *(m,3,3)* is only allowed, when the input coordinates
        comprise multiple models.

    Returns
    -------
    dihed : float or ndarray
        The dihedral angle(s) between the atoms. The shape is equal to
        the shape of the input `atoms` with the highest dimensionality
        minus the last axis.

    See Also
    --------
    index_dihedral : The same calculation, but using atom indices.
    dihedral_backbone : Calculate the dihedral angle along a peptide backbone.
    """
    v1 = displacement(atoms1, atoms2, box)
    v2 = displacement(atoms2, atoms3, box)
    v3 = displacement(atoms3, atoms4, box)
    norm_vector(v1)
    norm_vector(v2)
    norm_vector(v3)

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Calculation using atan2, to ensure the correct sign of the angle
    x = vector_dot(n1, n2)
    y = vector_dot(np.cross(n1, n2), v2)
    return np.arctan2(y, x)


def index_dihedral(*args, **kwargs):
    """
    index_dihedral(atoms, indices, periodic=False, box=None)

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
        :class:`ndarray`.
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

    See Also
    --------
    dihedral
    dihedral_backbone
    """
    return _call_non_index_function(dihedral, 4, *args, **kwargs)


def dihedral_backbone(atom_array):
    """
    Measure the characteristic backbone dihedral angles of a chain.

    Parameters
    ----------
    atom_array : AtomArray or AtomArrayStack
        The protein structure to measure the dihedral angles for.
        For missing backbone atoms the corresponding angles are `NaN`.

    Returns
    -------
    phi, psi, omega : ndarray
        An array containing the 3 backbone dihedral angles for every CA atom.
        `phi` is not defined at the N-terminus, `psi` and `omega` are not defined at the
        C-terminus.
        In these places the arrays have *NaN* values.
        If an :class:`AtomArrayStack` is given, the output angles are 2-dimensional,
        the first dimension corresponds to the model number.
    """
    amino_acid_mask = filter_amino_acids(atom_array)

    # Coordinates for dihedral angle calculation
    coord_n, coord_ca, coord_c = coord_for_atom_name_per_residue(
        atom_array,
        ("N", "CA", "C"),
        amino_acid_mask,
    )
    n_residues = coord_n.shape[-2]

    # Coordinates for dihedral angle calculation
    # Dim 0: Model index (only for atom array stacks)
    # Dim 1: Angle index
    # Dim 2: X, Y, Z coordinates
    # Dim 3: Atoms involved in dihedral angle
    if isinstance(atom_array, AtomArray):
        angle_coord_shape: tuple[int, ...] = (n_residues, 3, 4)
    elif isinstance(atom_array, AtomArrayStack):
        angle_coord_shape = (atom_array.stack_depth(), n_residues, 3, 4)
    coord_for_phi = np.full(angle_coord_shape, np.nan, dtype=np.float32)
    coord_for_psi = np.full(angle_coord_shape, np.nan, dtype=np.float32)
    coord_for_omg = np.full(angle_coord_shape, np.nan, dtype=np.float32)

    # fmt: off
    coord_for_phi[..., 1:,   :, 0] =  coord_c[..., 0:-1, :]
    coord_for_phi[..., 1:,   :, 1] =  coord_n[..., 1:,   :]
    coord_for_phi[..., 1:,   :, 2] = coord_ca[..., 1:,   :]
    coord_for_phi[..., 1:,   :, 3] =  coord_c[..., 1:,   :]

    coord_for_psi[..., 0:-1, :, 0] =  coord_n[..., 0:-1, :]
    coord_for_psi[..., 0:-1, :, 1] = coord_ca[..., 0:-1, :]
    coord_for_psi[..., 0:-1, :, 2] =  coord_c[..., 0:-1, :]
    coord_for_psi[..., 0:-1, :, 3] =  coord_n[..., 1:,   :]

    coord_for_omg[..., 0:-1, :, 0] = coord_ca[..., 0:-1, :]
    coord_for_omg[..., 0:-1, :, 1] =  coord_c[..., 0:-1, :]
    coord_for_omg[..., 0:-1, :, 2] =  coord_n[..., 1:,   :]
    coord_for_omg[..., 0:-1, :, 3] = coord_ca[..., 1:,   :]
    # fmt: on

    phi = dihedral(
        coord_for_phi[..., 0],
        coord_for_phi[..., 1],
        coord_for_phi[..., 2],
        coord_for_phi[..., 3],
    )
    psi = dihedral(
        coord_for_psi[..., 0],
        coord_for_psi[..., 1],
        coord_for_psi[..., 2],
        coord_for_psi[..., 3],
    )
    omg = dihedral(
        coord_for_omg[..., 0],
        coord_for_omg[..., 1],
        coord_for_omg[..., 2],
        coord_for_omg[..., 3],
    )

    return phi, psi, omg


def centroid(atoms):
    """
    Measure the centroid of a structure.

    Parameters
    ----------
    atoms : ndarray or AtomArray or AtomArrayStack
        The structures to determine the centroid from.
        Alternatively an ndarray containing the coordinates can be
        provided.

    Returns
    -------
    centroid : float or ndarray
        The centroid of the structure(s). :class:`ndarray` is returned when
        an :class:`AtomArrayStack` is given (centroid for each model).
    """
    return np.mean(coord(atoms), axis=-2)


def _call_non_index_function(
    function, expected_amount, atoms, indices, periodic=False, box=None
):
    """
    Call an `xxx()` function based on the parameters given to a
    `index_xxx()` function.
    """
    if indices.shape[-1] != expected_amount:
        raise ValueError(
            f"Expected length {expected_amount} in the last dimension "
            f"of the indices, but got length {indices.shape[-1]}"
        )
    coord_list = []
    for i in range(expected_amount):
        coord_list.append(coord(atoms)[..., indices[:, i], :])
    if periodic:
        if box is None:
            if isinstance(atoms, (AtomArray, AtomArrayStack)):
                box = atoms.box
            else:
                raise ValueError(
                    "If `atoms` are coordinates, the box must be set explicitly"
                )
    else:
        box = None
    return function(*coord_list, box)


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
                x = i * box[0, 0] + j * box[1, 0] + k * box[2, 0]
                y = i * box[0, 1] + j * box[1, 1] + k * box[2, 1]
                z = i * box[0, 2] + j * box[1, 2] + k * box[2, 2]
                periodic_shift.append([x, y, z])
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
        np.arange(len(shifted_diffs)), np.argmin(sq_distance, axis=1)
    ]
