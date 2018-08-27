# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides different transformations#
that can be applied on structures.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["translate", "rotate", "rotate_centered"]

import numpy as np
from .geometry import centroid


def translate(atoms, vector):
    """
    Translate a list of atoms by a given vector.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack
        The atoms whose coordinates are altered.
    vector: array-like, length=3
        The translation vector :math:`(x, y, z)`.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack
        A copy of the input atoms, translated by the given vector.
    """
    if len(vector) != 3:
        raise ValueError("Translation vector must be container of length 3")
    transformed = atoms.copy()
    transformed.coord += np.array(vector)
    return transformed

def rotate(atoms, angles):
    """
    Rotates a list of atoms by given angles.
    
    The rotations are centered at the origin and are performed sequentially
    in the order x,y,z.
    
    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack
        The atoms whose coordinates are altered.
    angles: array-like, length=3
        The rotation angles in radians around x, y and z.
    
    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack
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
    transformed = atoms.copy()
    transformed.coord = np.dot(transformed.coord, rot_x)
    transformed.coord = np.dot(transformed.coord, rot_y)
    transformed.coord = np.dot(transformed.coord, rot_z)
    return transformed

def rotate_centered(atoms, angles):
    """
    Rotates a list of atoms by given angles.
    
    The rotations are centered at the centroid of the corresponding
    structure and are performed sequentially in the order x,y,z.
    
    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms whose coordinates are altered.
    angles: array-like, length=3
        the rotation angles in radians around axes x, y and z.
    
    Returns
    -------
    transformed : AtomArray or AtomArrayStack
        A copy of the input atoms, rotated by the given angles.
        
    See Also
    --------
    rotate
    """
    # Rotation around centroid requires translation of centroid to origin
    transformed = atoms.copy()
    centro = centroid(transformed)
    transformed.coord -= centro
    transformed = rotate(transformed, angles)
    transformed.coord += centro
    return transformed