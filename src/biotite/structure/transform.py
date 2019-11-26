# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides different transformations#
that can be applied on structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["translate", "rotate", "rotate_centered"]

import numpy as np
from .geometry import centroid
from .error import BadStructureError
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import norm_vector, vector_dot


def translate(atoms, vector):
    """
    Translate a list of atoms by a given vector.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms whose coordinates are altered.
        The coordinates can be directly provided as :class:`ndarray`.
    vector: array-like, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The translation vector :math:`(x, y, z)`.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms, translated by the given vector.
    """
    positions = coord(atoms).copy()
    vector = np.asarray(vector)
    
    if vector.shape[-1] != 3:
        raise ValueError("Translation vector must contain 3 coordinates")
    positions += vector
    return _put_back(atoms, positions)


def rotate(atoms, angles):
    """
    Rotates a list of atoms by given angles.
    
    The rotations are centered at the origin and are performed sequentially
    in the order x,y,z.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms whose coordinates are altered.
        The coordinates can be directly provided as :class:`ndarray`.
    angles: array-like, length=3
        The rotation angles in radians around x, y and z.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms, rotated by the given angles.
        
    See Also
    --------
    rotate_centered
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
    
    # Copy AtomArray(Stack) and apply rotations
    # Note that the coordinates are treated as row vector
    positions = coord(atoms).copy()
    positions = np.dot(positions, rot_x)
    positions = np.dot(positions, rot_y)
    positions = np.dot(positions, rot_z)
    
    return _put_back(atoms, positions)


def rotate_centered(atoms, angles):
    """
    Rotates a list of atoms by given angles.
    
    The rotations are centered at the centroid of the corresponding
    structure and are performed sequentially in the order x,y,z.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms whose coordinates are altered.
        The coordinates can be directly provided as :class:`ndarray`.
    angles: array-like, length=3
        the rotation angles in radians around axes x, y and z.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms, rotated by the given angles.
        
    See Also
    --------
    rotate
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
    rot_coord = coord(atoms).copy()
    if support is not None:
        # Transform coordinates
        # so that the axis support vector is at (0,0,0)
        rot_coord -= np.asarray(support)
    
    # Normalize axis
    axis = np.asarray(axis, dtype=np.float32).copy()
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
        [cos_a + icos_a*x**2,   icos_a*x*y - z*sin_a,  icos_a*x*z + y*sin_a],
        [icos_a*x*y + z*sin_a,  cos_a + icos_a*y**2,   icos_a*y*z - x*sin_a],
        [icos_a*x*z - y*sin_a,  icos_a*y*z + x*sin_a,  cos_a + icos_a*z**2 ]
    ])
    # Apply rotation
    rot_coord = np.dot(rot_matrix, rot_coord.T).T

    if support is not None:
        # Transform coordinates back to original support vector position
        rot_coord += np.asarray(support)
    
    return _put_back(atoms, rot_coord)


def align_vectors(atoms, source_direction, target_direction,
                  source_position=None, target_position=None):
    source_direction = np.asarray(source_direction, dtype=np.float32)
    target_direction = np.asarray(target_direction, dtype=np.float32)
    if source_position is not None: 
        source_position = np.asarray(source_position, dtype=np.float32)
    if target_position is not None: 
        target_position = np.asarray(target_position, dtype=np.float32)

    moved_coord = coord(atoms).copy()
    if source_position is not None:
        # Transform coordinates
        # so that the position of the source vector is at (0,0,0)
        moved_coord -= source_position
    
    # Normalize direction vectors
    source_direction = source_direction.copy()
    norm_vector(source_direction)
    target_direction = target_direction.copy()
    norm_vector(target_direction)
    # Formula is taken from
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    v = np.cross(source_direction, target_direction)
    v_c = np.array([
        [ 0,         -v[..., 2],  v[..., 1]],
        [ v[..., 2],  0,         -v[..., 0]],
        [-v[..., 1],  v[..., 0],  0        ]
    ], dtype=float)
    cos_a = vector_dot(source_direction, target_direction)
    if np.all(cos_a == -1):
        raise ValueError(
            "Direction vectors are point into opposite directions, "
            "cannot calculate rotation matrix"
        )
    rot_matrix = np.identity(3) + v_c + np.matmul(v_c, v_c) / (1+cos_a)
    # Apply rotation
    moved_coord = np.dot(rot_matrix, moved_coord.T).T
    
    if target_position is not None:
        # Transform coordinates to position of the target vector
        moved_coord += target_position
    
    return _put_back(atoms, moved_coord)


def _put_back(input_atoms, transformed):
    if isinstance(input_atoms, (Atom, AtomArray, AtomArrayStack)):
        moved_atoms = input_atoms.copy()
        moved_atoms.coord = transformed
        return moved_atoms
    else:
        return transformed