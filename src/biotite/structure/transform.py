# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides different transformations#
that can be applied on structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann", "Claude J. Rogers"
__all__ = [
    "AffineTransformation",
    "translate",
    "rotate",
    "rotate_centered",
    "rotate_about_axis",
    "orient_principal_components",
    "align_vectors",
]

import numpy as np
from biotite.structure.atoms import Atom, AtomArray, AtomArrayStack, coord
from biotite.structure.util import matrix_rotate, norm_vector, vector_dot


class AffineTransformation:
    """
    An affine transformation, consisting of translations and a rotation.

    Parameters
    ----------
    center_translation : ndarray, shape=(3,) or shape=(m,3), dtype=float
        The translation vector for moving the centroid into the
        origin.
    rotation : ndarray, shape=(3,3) or shape=(m,3,3), dtype=float
        The rotation matrix.
    target_translation : ndarray, shape=(m,3), dtype=float
        The translation vector for moving the structure onto the
        fixed one.

    Attributes
    ----------
    center_translation, rotation, target_translation : ndarray
        Same as the parameters.
        The dimensions are always expanded to *(m,3)* or *(m,3,3)*,
        respectively.
    """

    def __init__(self, center_translation, rotation, target_translation):
        self.center_translation = _expand_dims(center_translation, 2)
        self.rotation = _expand_dims(rotation, 3)
        self.target_translation = _expand_dims(target_translation, 2)

    def apply(self, atoms):
        """
        Apply this transformation on the given structure.

        Parameters
        ----------
        atoms : AtomArray or AtomArrayStack or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
            The structure to apply the transformation on.

        Returns
        -------
        transformed : AtomArray or AtomArrayStack or ndarray, shape(n,), dtype=float or ndarray, shape(m,n), dtype=float
            A copy of the `atoms` structure, with transformations applied.
            Only coordinates are returned, if coordinates were given in `atoms`.

        Examples
        --------

        >>> coord = np.arange(15).reshape(5,3)
        >>> print(coord)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]
         [12 13 14]]
        >>> # Rotates 90 degrees around the z-axis
        >>> transform = AffineTransformation(
        ...     center_translation=np.array([0,0,0]),
        ...     rotation=np.array([
        ...         [0, -1,  0],
        ...         [1,  0,  0],
        ...         [0,  0,  1]
        ...     ]),
        ...     target_translation=np.array([0,0,0])
        ... )
        >>> print(transform.apply(coord))
        [[ -1.   0.   2.]
         [ -4.   3.   5.]
         [ -7.   6.   8.]
         [-10.   9.  11.]
         [-13.  12.  14.]]
        """
        mobile_coord = coord(atoms)
        original_shape = mobile_coord.shape
        mobile_coord = _reshape_to_3d(mobile_coord)
        if (
            self.rotation.shape[0] != 1
            and mobile_coord.shape[0] != self.rotation.shape[0]
        ):
            raise IndexError(
                f"Number of transformations is {self.rotation.shape[0]}, "
                f"but number of structure models is {mobile_coord.shape[0]}"
            )

        superimposed_coord = mobile_coord.copy()
        superimposed_coord += self.center_translation[:, np.newaxis, :]
        superimposed_coord = _multi_matmul(self.rotation, superimposed_coord)
        superimposed_coord += self.target_translation[:, np.newaxis, :]

        superimposed_coord = superimposed_coord.reshape(original_shape)
        if isinstance(atoms, np.ndarray):
            return superimposed_coord
        else:
            superimposed = atoms.copy()
            superimposed.coord = superimposed_coord
            return superimposed

    def as_matrix(self):
        """
        Get the translations and rotation as a combined 4x4
        transformation matrix.

        Multiplying this matrix with coordinates in the form
        *(x, y, z, 1)* will apply the same transformation as
        :meth:`apply()` to coordinates in the form *(x, y, z)*.

        Returns
        -------
        transformation_matrix : ndarray, shape=(m,4,4), dtype=float
            The transformation matrix.
            *m* is the number of models in the transformation.

        Examples
        --------

        >>> coord = np.arange(15).reshape(5,3)
        >>> print(coord)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]
         [12 13 14]]
        >>> # Rotates 90 degrees around the z-axis
        >>> transform = AffineTransformation(
        ...     center_translation=np.array([0,0,0]),
        ...     rotation=np.array([
        ...         [0, -1,  0],
        ...         [1,  0,  0],
        ...         [0,  0,  1]
        ...     ]),
        ...     target_translation=np.array([0,0,0])
        ... )
        >>> print(transform.apply(coord))
        [[ -1.   0.   2.]
         [ -4.   3.   5.]
         [ -7.   6.   8.]
         [-10.   9.  11.]
         [-13.  12.  14.]]
        >>> # Use a 4x4 matrix for transformation as alternative
        >>> coord_4 = np.concatenate([coord, np.ones((len(coord), 1))], axis=-1)
        >>> print(coord_4)
        [[ 0.  1.  2.  1.]
         [ 3.  4.  5.  1.]
         [ 6.  7.  8.  1.]
         [ 9. 10. 11.  1.]
         [12. 13. 14.  1.]]
        >>> print((transform.as_matrix()[0] @ coord_4.T).T)
        [[ -1.   0.   2.   1.]
         [ -4.   3.   5.   1.]
         [ -7.   6.   8.   1.]
         [-10.   9.  11.   1.]
         [-13.  12.  14.   1.]]
        """
        n_models = self.rotation.shape[0]
        rotation_mat = _3d_identity(n_models, 4)
        rotation_mat[:, :3, :3] = self.rotation
        center_translation_mat = _3d_identity(n_models, 4)
        center_translation_mat[:, :3, 3] = self.center_translation
        target_translation_mat = _3d_identity(n_models, 4)
        target_translation_mat[:, :3, 3] = self.target_translation
        return target_translation_mat @ rotation_mat @ center_translation_mat

    def __eq__(self, other):
        if not isinstance(other, AffineTransformation):
            return False
        if not np.array_equal(self.center_translation, other.center_translation):
            return False
        if not np.array_equal(self.rotation, other.rotation):
            return False
        if not np.array_equal(self.target_translation, other.target_translation):
            return False
        return True


def _expand_dims(array, n_dims):
    """
    Expand the dimensions of an `ndarray` to a certain number of
    dimensions.
    """
    while array.ndim < n_dims:
        array = array[np.newaxis, ...]
    return array


def _3d_identity(m, n):
    """
    Create an array of *m* identity matrices of shape *(n, n)*
    """
    matrices = np.zeros((m, n, n), dtype=float)
    indices = np.arange(n)
    matrices[:, indices, indices] = 1
    return matrices


def _reshape_to_3d(coord):
    """
    Reshape the coordinate array to 3D, if it is 2D.
    """
    if coord.ndim < 2:
        raise ValueError("Coordinates must be at least two-dimensional")
    if coord.ndim == 2:
        return coord[np.newaxis, ...]
    elif coord.ndim == 3:
        return coord
    else:
        raise ValueError("Coordinates must be at most three-dimensional")


def translate(atoms, vector):
    """
    Translate the given atoms or coordinates by a given vector.

    Parameters
    ----------
    atoms : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    vector : array-like, shape=(3,) or shape=(n,3) or shape=(m,n,3)
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
    angles : array-like, length=3
        The rotation angles in radians around *x*, *y* and *z*.

    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated by the given
        angles.

    See Also
    --------
    rotate_centered : Rotate atoms about the centroid.
    rotate_about_axis : Rotate atoms about a given axis.

    Examples
    --------
    Rotation about the *z*-axis by 90 degrees:

    >>> position = np.array([2.0, 0.0, 0.0])
    >>> rotated = rotate(position, angles=(0, 0, 0.5*np.pi))
    >>> print(rotated)
    [1.225e-16 2.000e+00 0.000e+00]
    """
    from numpy import cos, sin

    # Check if "angles" contains 3 angles for all dimensions
    if len(angles) != 3:
        raise ValueError("Translation vector must be container of length 3")
    # Create rotation matrices for all 3 dimensions
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, cos(angles[0]), -sin(angles[0])],
            [0, sin(angles[0]), cos(angles[0])],
        ]
    )

    rot_y = np.array(
        [
            [cos(angles[1]), 0, sin(angles[1])],
            [0, 1, 0],
            [-sin(angles[1]), 0, cos(angles[1])],
        ]
    )

    rot_z = np.array(
        [
            [cos(angles[2]), -sin(angles[2]), 0],
            [sin(angles[2]), cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )

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
    angles : array-like, length=3
        The rotation angles in radians around axes *x*, *y* and *z*.

    Returns
    -------
    transformed : Atom or AtomArray or AtomArrayStack or ndarray, shape=(3,) or shape=(n,3) or shape=(m,n,3)
        A copy of the input atoms or coordinates, rotated by the given
        angles.

    See Also
    --------
    rotate : Rotate atoms about the origin.
    rotate_about_axis : Rotate atoms about a given axis.
    """
    # Avoid circular import
    from biotite.structure.geometry import centroid

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
    rotate : Rotate atoms about the origin.
    rotate_centered : Rotate atoms about the centroid.

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
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    # Rotation matrix is taken from
    # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot_matrix = np.array(
        [
            [
                cos_a + icos_a * x**2,
                icos_a * x * y - z * sin_a,
                icos_a * x * z + y * sin_a,
            ],
            [
                icos_a * x * y + z * sin_a,
                cos_a + icos_a * y**2,
                icos_a * y * z - x * sin_a,
            ],
            [
                icos_a * x * z - y * sin_a,
                icos_a * y * z + x * sin_a,
                cos_a + icos_a * z**2,
            ],
        ]
    )

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


def orient_principal_components(atoms, order=None):
    """
    Translate and rotate the atoms to be centered at the origin with
    the principal axes aligned to the Cartesian axes, as specified by
    the `order` parameter. By default, x, y, z.

    By default, the resulting coordinates have the highest variance in
    the x-axis and the lowest variance on the z-axis. Setting the `order`
    parameter will change the direction of the highest variance.
    For example, ``order=(2, 1, 0)`` results in highest variance along the
    z-axis and lowest along the x-axis.

    Parameters
    ----------
    atoms : AtomArray or ndarray, shape=(n,3)
        The atoms of which the coordinates are transformed.
        The coordinates can be directly provided as :class:`ndarray`.
    order : array-like, length=3
        The order of decreasing variance. Setting `order` to ``(2, 0, 1)``
        results in highest variance along the y-axis and lowest along
        the x-axis. Default = ``(0, 1, 2)``.

    Returns
    -------
    AtomArray or ndarray, shape=(n,3)
        The atoms with coordinates centered at the orgin and aligned with
        xyz axes.

    Examples
    --------
    Align principal components to xyz axes (default), or specify the order
    of variance.

    >>> print("original variance =", atom_array.coord.var(axis=0))
    original variance = [26.517 20.009  9.325]
    >>> moved = orient_principal_components(atom_array)
    >>> print("moved variance =", moved.coord.var(axis=0))
    moved variance = [28.906 18.495  8.450]
    >>> # Note the increase in variance along the x-axis
    >>> # Specifying the order keyword changes the orientation
    >>> moved_z = orient_principal_components(atom_array, order=(2, 1, 0))
    >>> print("moved (zyx) variance =", moved_z.coord.var(axis=0))
    moved (zyx) variance = [ 8.450 18.495 28.906]
    """
    # Check user input
    coords = coord(atoms)
    if coords.ndim != 2:
        raise ValueError(f"Expected input shape of (n, 3), got {coords.shape}.")
    row, col = coords.shape
    if (row < 3) or (col != 3):
        raise ValueError(
            f"Expected at least 3 entries, {row} given, and 3 dimensions, {col} given."
        )
    if order is None:
        order = np.array([0, 1, 2])
    else:
        order = np.asarray(order, dtype=int)
        if order.shape != (3,):
            raise ValueError(f"Expected order to have shape (3,), not {order.shape}")
        if not (np.sort(order) == np.arange(3)).all():
            raise ValueError("Expected order to contain [0, 1, 2].")

    # place centroid of the atoms at the origin
    centered = coords - coords.mean(axis=0)

    # iterate a few times to ensure the ideal rotation has been applied
    identity = np.eye(3)
    MAX_ITER = 50
    for _ in range(MAX_ITER):
        # PCA, W is the component matrix, s ~ explained variance
        _, sigma, components = np.linalg.svd(centered)

        # sort 1st, 2nd, 3rd pca compontents along x, y, z
        idx = sigma.argsort()[::-1][order]
        ident = np.eye(3)[:, idx]

        # Run the Kabsch algorithm to orient principal components
        # along the x, y, z axes
        v, _, wt = np.linalg.svd(components.T @ ident)

        if np.linalg.det(v) * np.linalg.det(wt) < -0:
            v[:, -1] *= -1
        # Calculate rotation matrix
        rotation = v @ wt
        if np.isclose(rotation, identity, atol=1e-5).all():
            break
        # Apply rotation, keep molecule centered on the origin
        centered = centered @ rotation
    return _put_back(atoms, centered)


def align_vectors(
    atoms,
    origin_direction,
    target_direction,
    origin_position=None,
    target_position=None,
):
    """
    Apply a transformation to atoms or coordinates, that would transfer
    a origin vector to a target vector.

    At first the transformation (translation and rotation), that is
    necessary to align the origin vector to the target vector is
    calculated.
    This means, that the application of the transformation on the
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
        A copy of the input atoms or coordinates with the applied
        transformation.

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
    origin_direction = np.asarray(origin_direction, dtype=np.float32).squeeze()
    target_direction = np.asarray(target_direction, dtype=np.float32).squeeze()
    # check that original and target direction are vectors of shape (3,)
    if origin_direction.shape != (3,):
        raise ValueError(
            f"Expected origin vector to have shape (3,), got {origin_direction.shape}"
        )
    if target_direction.shape != (3,):
        raise ValueError(
            f"Expected target vector to have shape (3,), got {target_direction.shape}"
        )
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
    vx, vy, vz = np.cross(origin_direction, target_direction)
    v_c = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]], dtype=float)
    cos_a = vector_dot(origin_direction, target_direction)
    if np.all(cos_a == -1):
        raise ValueError(
            "Direction vectors are point into opposite directions, "
            "cannot calculate rotation matrix"
        )
    rot_matrix = np.identity(3) + v_c + (v_c @ v_c) / (1 + cos_a)

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


def _multi_matmul(matrices, vectors):
    """
    Calculate the matrix multiplication of m matrices
    with m x n vectors.
    """
    return np.transpose(
        np.matmul(matrices, np.transpose(vectors, axes=(0, 2, 1))), axes=(0, 2, 1)
    )
