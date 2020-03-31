# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides different transformations#
that can be applied on structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["translate", "rotate", "rotate_centered", "rotate_about_axis",
           "align_vectors"]

import numpy as np
from .geometry import centroid
from .error import BadStructureError
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import norm_vector, vector_dot, matrix_rotate


def translate(atoms, vector):
    """
    Translate the given atoms or coordinates by a given vector.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    vector: array-like, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The translation vector :math:`(x, y, z)`.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, translated by the
        given vector.
    """
    positions = coord(atoms).copy()
    vector = np.asarray(vector)
    
    if vector.shape[-1] != 3:
        raise ValueError("Translation vector must contain 3 coordinates")
    positions += vector
    return _put_back(atoms, positions)


def rotate(atoms, angles):
    """
    Rotate the given atoms or coordinates about the *x*, *y* and *z*
    axes by given angles.
    
    The rotations are centered at the origin and are performed
    sequentially in the order *x*, *y*, *z*.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    angles: array-like, length=3
        The rotation angles in radians around *x*, *y* and *z*.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated by the given
        angles.
        
    See Also
    --------
    rotate_centered
    rotate_about_axis

    Examples
    --------
    Rotation about the *z*-axis by 90 degrees:

    >>> position = np.array([2.0, 0.0, 0.0])
    >>> rotated = rotate(position, angles=(0, 0, 0.5*np.pi))
    >>> print(rotated)
    [1.225e-16 2.000e+00 0.000e+00]
    """
    from numpy import sin, cos

    # Check if "angles" contains 3 angles for all dimensions
    if len(angles) != 3:
        raise ValueError("Translation vector must be container of length 3")
    # Create rotation matrices for all 3 dimensions
    rot_x = np.array([[ 1,               0,               0               ],
                      [ 0,               cos(angles[0]),  -sin(angles[0]) ],
                      [ 0,               sin(angles[0]),  cos(angles[0])  ]])
    
    rot_y = np.array([[ cos(angles[1]),  0,               sin(angles[1])  ],
                      [ 0,               1,               0               ],
                      [ -sin(angles[1]), 0,               cos(angles[1])  ]])
    
    rot_z = np.array([[ cos(angles[2]),  -sin(angles[2]), 0               ],
                      [ sin(angles[2]),  cos(angles[2]),  0               ],
                      [ 0,               0,               1               ]])
    
    positions = coord(atoms).copy()
    positions = matrix_rotate(positions, rot_z @ rot_y @ rot_x)
    
    return _put_back(atoms, positions)


def rotate_centered(atoms, angles):
    """
    Rotate the given atoms or coordinates about the *x*, *y* and *z*
    axes by given angles.
    
    The rotations are centered at the centroid of the corresponding
    structure and are performed sequentially in the order *x*, *y*, *z*.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    angles: array-like, length=3
        The rotation angles in radians around axes *x*, *y* and *z*.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated by the given
        angles.
        
    See Also
    --------
    rotate
    rotate_about_axis
    """
    if len(coord(atoms).shape) == 1:
        # Single value -> centered rotation does not change coordinates
        return atoms.copy()
    
    # Rotation around centroid requires moving centroid to origin
    center = coord(centroid(atoms))
    # 'centroid()' removes the second last dimesion
    # -> add dimension again to ensure correct translation
    center = center[..., np.newaxis, :]
    translated = translate(atoms, -center)
    rotated = rotate(translated, angles)
    translated_back = translate(rotated, center)
    return translated_back


def rotate_about_axis(atoms, axis, angle, support=None):
    """
    Rotate the given atoms or coordinates about a given axis by a given
    angle.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    axis : array-like, length=3
        A vector representing the direction of the rotation axis.
        The length of the vector is irrelevant.
    angle : float
        The rotation angle in radians.
    support : array-like, length=3, optional
        An optional support vector for the rotation axis, i.e. the
        center of the rotation.
        By default, the center of the rotation is at *(0,0,0)*.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated about the
        given axis.
        
    See Also
    --------
    rotate
    rotate_centered

    Examples
    --------
    Rotation about a custom axis on the *y*-*z*-plane by 90 degrees:

    >>> position = np.array([2.0, 0.0, 0.0])
    >>> axis = [0.0, 1.0, 1.0]
    >>> rotated = rotate_about_axis(position, axis, angle=0.5*np.pi)
    >>> print(rotated)
    [ 1.225e-16  1.414e+00 -1.414e+00]
    """
    positions = coord(atoms).copy()
    if support is not None:
        # Transform coordinates
        # so that the axis support vector is at (0,0,0)
        positions -= np.asarray(support)
    
    # Normalize axis
    axis = np.asarray(axis, dtype=np.float32).copy()
    if np.linalg.norm(axis) == 0:
        raise ValueError("Length of the rotation axis is 0")
    norm_vector(axis)
    # Save some interim values, that are used repeatedly in the
    # rotation matrix, for clarity and to save computation time
    sin_a = np.sin(angle)
    cos_a = np.cos(angle)
    icos_a = 1 - cos_a
    x = axis[...,0]
    y = axis[...,1]
    z = axis[...,2]
    # Rotation matrix is taken from
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot_matrix = np.array([
        [ cos_a + icos_a*x**2,  icos_a*x*y - z*sin_a,  icos_a*x*z + y*sin_a],
        [icos_a*x*y + z*sin_a,   cos_a + icos_a*y**2,  icos_a*y*z - x*sin_a],
        [icos_a*x*z - y*sin_a,  icos_a*y*z + x*sin_a,   cos_a + icos_a*z**2]
    ])

    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = positions.ndim
    if orig_ndim > 2:
        orig_shape = positions.shape
        positions = positions.reshape(-1, 3)
    # Apply rotation
    positions = np.dot(rot_matrix, positions.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        positions = positions.reshape(*orig_shape)

    if support is not None:
        # Transform coordinates back to original support vector position
        positions += np.asarray(support)
    
    return _put_back(atoms, positions)


def align_vectors(atoms, origin_direction, target_direction,
                  origin_position=None, target_position=None):
    """
    Apply a transformation to atoms or coordinates, that would transfer
    a origin vector to a target vector.

    At first the transformation (translation and rotation), that is
    necessary to align the origin vector to the target vector is
    calculated.
    This means means, that the application of the transformation on the
    origin vector would give the target vector.
    Then the same transformation is applied to the given
    atoms/coordinates. 
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    origin_direction, target_direction : array-like, length=3
        The vectors representing the direction of the origin or target,
        respectively.
        The length of the vectors is irrelevant.
    origin_position, target_position : array-like, length=3, optional
        Optional support vectors for the origin or target, respectively.
        By default, origin and target start at *(0,0,0)*.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated about the
        given axis.
        
    See Also
    --------
    rotate
    rotate_centered

    Examples
    --------
    Align two different residues at their CA-CB bond, i.e. the CA and CB
    position of both residues should be equal:

    >>> first_residue = atom_array[atom_array.res_id == 1]
    >>> second_residue = atom_array[atom_array.res_id == 2]
    >>> first_residue_ca = first_residue[first_residue.atom_name == "CA"]
    >>> first_residue_cb = first_residue[first_residue.atom_name == "CB"]
    >>> second_residue_ca = second_residue[second_residue.atom_name == "CA"]
    >>> second_residue_cb = second_residue[second_residue.atom_name == "CB"]
    >>> first_residue = align_vectors(
    ...     first_residue,
    ...     origin_direction = first_residue_cb.coord - first_residue_ca.coord,
    ...     target_direction = second_residue_cb.coord - second_residue_ca.coord,
    ...     origin_position = first_residue_ca.coord,
    ...     target_position = second_residue_ca.coord
    ... )
    >>> # The CA and CB coordinates of both residues are now almost equal
    >>> print(first_residue)
        A       1  ASN N      N        -5.549    3.788   -1.125
        A       1  ASN CA     C        -4.923    4.002   -2.452
        A       1  ASN C      C        -3.877    2.949   -2.808
        A       1  ASN O      O        -4.051    2.292   -3.825
        A       1  ASN CB     C        -4.413    5.445   -2.618
        A       1  ASN CG     C        -5.593    6.413   -2.670
        A       1  ASN OD1    O        -6.738    5.988   -2.651
        A       1  ASN ND2    N        -5.362    7.711   -2.695
        A       1  ASN H1     H        -5.854    2.830   -1.023
        A       1  ASN H2     H        -4.902    4.017   -0.382
        A       1  ASN H3     H        -6.357    4.397   -1.047
        A       1  ASN HA     H        -5.711    3.870   -3.198
        A       1  ASN HB2    H        -3.781    5.697   -1.788
        A       1  ASN HB3    H        -3.860    5.526   -3.556
        A       1  ASN HD21   H        -4.440    8.115   -2.680
        A       1  ASN HD22   H        -6.190    8.284   -2.750
    >>> print(second_residue)
        A       2  LEU N      N        -6.379    4.031   -2.228
        A       2  LEU CA     C        -4.923    4.002   -2.452
        A       2  LEU C      C        -4.136    3.187   -1.404
        A       2  LEU O      O        -3.391    2.274   -1.760
        A       2  LEU CB     C        -4.411    5.450   -2.619
        A       2  LEU CG     C        -4.795    6.450   -1.495
        A       2  LEU CD1    C        -3.612    6.803   -0.599
        A       2  LEU CD2    C        -5.351    7.748   -2.084
        A       2  LEU H      H        -6.821    4.923   -2.394
        A       2  LEU HA     H        -4.750    3.494   -3.403
        A       2  LEU HB2    H        -3.340    5.414   -2.672
        A       2  LEU HB3    H        -4.813    5.817   -3.564
        A       2  LEU HG     H        -5.568    6.022   -0.858
        A       2  LEU HD11   H        -3.207    5.905   -0.146
        A       2  LEU HD12   H        -2.841    7.304   -1.183
        A       2  LEU HD13   H        -3.929    7.477    0.197
        A       2  LEU HD21   H        -4.607    8.209   -2.736
        A       2  LEU HD22   H        -6.255    7.544   -2.657
        A       2  LEU HD23   H        -5.592    8.445   -1.281
    """
    origin_direction = np.asarray(origin_direction, dtype=np.float32)
    target_direction = np.asarray(target_direction, dtype=np.float32)
    if np.linalg.norm(origin_direction) == 0:
        raise ValueError("Length of the origin vector is 0")
    if np.linalg.norm(target_direction) == 0:
        raise ValueError("Length of the target vector is 0")
    if origin_position is not None: 
        origin_position = np.asarray(origin_position, dtype=np.float32)
    if target_position is not None: 
        target_position = np.asarray(target_position, dtype=np.float32)

    positions = coord(atoms).copy()
    if origin_position is not None:
        # Transform coordinates
        # so that the position of the origin vector is at (0,0,0)
        positions -= origin_position
    
    # Normalize direction vectors
    origin_direction = origin_direction.copy()
    norm_vector(origin_direction)
    target_direction = target_direction.copy()
    norm_vector(target_direction)
    # Formula is taken from
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    v = np.cross(origin_direction, target_direction)
    v_c = np.array([
        [         0, -v[..., 2],  v[..., 1]],
        [ v[..., 2],          0, -v[..., 0]],
        [-v[..., 1],  v[..., 0],          0]
    ], dtype=float)
    cos_a = vector_dot(origin_direction, target_direction)
    if np.all(cos_a == -1):
        raise ValueError(
            "Direction vectors are point into opposite directions, "
            "cannot calculate rotation matrix"
        )
    rot_matrix = np.identity(3) + v_c + (v_c @ v_c) / (1+cos_a)
    
    positions = matrix_rotate(positions, rot_matrix)
    
    if target_position is not None:
        # Transform coordinates to position of the target vector
        positions += target_position

    return _put_back(atoms, positions)


def _put_back(input_atoms, transformed):
    """
    Put the altered coordinates back into the :class:`Atom`,
    :class:`AtomArray` or :class:`AtomArrayStack`, if the input was one
    of these types.
    """
    if isinstance(input_atoms, (Atom, AtomArray, AtomArrayStack)):
        moved_atoms = input_atoms.copy()
        moved_atoms.coord = transformed
        return moved_atoms
    else:
        return transformed