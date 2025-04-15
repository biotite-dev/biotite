# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions related to working with the simulation box or unit cell
of a structure.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "space_group_transforms",
    "vectors_from_unitcell",
    "unitcell_from_vectors",
    "box_volume",
    "repeat_box",
    "repeat_box_coord",
    "move_inside_box",
    "remove_pbc",
    "remove_pbc_from_coord",
    "coord_to_fraction",
    "fraction_to_coord",
    "is_orthogonal",
]

import functools
import json
from numbers import Integral
from pathlib import Path
import numpy as np
import numpy.linalg as linalg
from biotite.structure.atoms import repeat
from biotite.structure.chains import get_chain_masks, get_chain_starts
from biotite.structure.error import BadStructureError
from biotite.structure.molecules import get_molecule_masks
from biotite.structure.transform import AffineTransformation
from biotite.structure.util import vector_dot


def space_group_transforms(space_group):
    """
    Get the coordinate transformations for a given space group.

    Applying each transformation to a structure (in fractional coordinates) reproduces
    the entire unit cell.

    Parameters
    ----------
    space_group : str or int
        The space group name (full *Hermann-Mauguin* symbol) or
        *International Table*'s number.

    Returns
    -------
    transformations : list of AffineTransformation
        The transformations that creates the symmetric copies of a structure in a unit
        cell of the given space group.
        Note that the transformations need to be applied to coordinates in fractions
        of the unit cell and also return fractional coordinates, when applied.

    See Also
    --------
    coord_to_fraction : Used to convert to fractional coordinates.
    fraction_to_coord : Used to convert back to Cartesian coordinates.

    Examples
    --------

    >>> transforms = space_group_transforms("P 21 21 21")
    >>> for transform in transforms:
    ...     print(transform.rotation)
    ...     print(transform.target_translation)
    ...     print()
    [[[1. 0. 0.]
      [0. 1. 0.]
      [0. 0. 1.]]]
    [[0. 0. 0.]]
    <BLANKLINE>
    [[[-1.  0.  0.]
      [ 0. -1.  0.]
      [ 0.  0.  1.]]]
    [[0.5 0.0 0.5]]
    <BLANKLINE>
    [[[-1.  0.  0.]
      [ 0.  1.  0.]
      [ 0.  0. -1.]]]
    [[0.0 0.5 0.5]]
    <BLANKLINE>
    [[[ 1.  0.  0.]
      [ 0. -1.  0.]
      [ 0.  0. -1.]]]
    [[0.5 0.5 0.0]]
    <BLANKLINE>

    Reproduce the unit cell for some coordinates (in this case only one atom).

    >>> asym_coord = np.array([[1.0, 2.0, 3.0]])
    >>> box = np.eye(3) * 10
    >>> transforms = space_group_transforms("P 21 21 21")
    >>> # Apply the transformations to fractional coordinates of the asymmetric unit
    >>> unit_cell = np.concatenate(
    ...     [
    ...         fraction_to_coord(transform.apply(coord_to_fraction(asym_coord, box)), box)
    ...         for transform in transforms
    ...     ]
    ... )
    >>> print(unit_cell)
    [[ 1.  2.  3.]
     [ 4. -2.  8.]
     [-1.  7.  2.]
     [ 6.  3. -3.]]
    """
    transformation_data = _get_transformation_data()

    if isinstance(space_group, str):
        try:
            space_group_index = transformation_data["group_names"][space_group]
        except KeyError:
            raise ValueError(f"Space group '{space_group}' does not exist")
    else:
        try:
            space_group_index = transformation_data["group_numbers"][str(space_group)]
        except KeyError:
            raise ValueError(f"Space group number {space_group} does not exist")

    space_group = transformation_data["space_groups"][space_group_index]
    transformations = []
    for transformation_index in space_group:
        matrix = np.zeros((3, 3), dtype=np.float32)
        translation = np.zeros(3, dtype=np.float32)
        for i, part_index in enumerate(
            transformation_data["transformations"][transformation_index]
        ):
            part = transformation_data["transformation_parts"][part_index]
            matrix[i, :] = part[:3]
            translation[i] = part[3]
        transformations.append(
            AffineTransformation(
                center_translation=np.zeros(3, dtype=np.float32),
                rotation=matrix,
                target_translation=translation,
            )
        )
    return transformations


def vectors_from_unitcell(len_a, len_b, len_c, alpha, beta, gamma):
    """
    Calculate the three vectors spanning a box from the unit cell
    lengths and angles.

    The return value of this function are the three box vectors as
    required for the :attr:`box` attribute in atom arrays and stacks.

    Parameters
    ----------
    len_a, len_b, len_c : float
        The lengths of the three box/unit cell vectors *a*, *b* and *c*.
    alpha, beta, gamma : float
        The angles between the box vectors in radians.
        *alpha* is the angle between *b* and *c*,
        *beta* between *a* and *c*, *gamma* between *a* and *b*.

    Returns
    -------
    box : ndarray, dtype=float, shape=(3,3)
        The three box vectors.
        The vector components are in the last dimension.
        The value can be directly used as :attr:`box` attribute in an
        atom array.

    See Also
    --------
    unitcell_from_vectors : The reverse operation.
    """
    a_x = len_a
    b_x = len_b * np.cos(gamma)
    b_y = len_b * np.sin(gamma)
    c_x = len_c * np.cos(beta)
    c_y = len_c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    c_z = np.sqrt(len_c * len_c - c_x * c_x - c_y * c_y)
    box = np.array([[a_x, 0, 0], [b_x, b_y, 0], [c_x, c_y, c_z]], dtype=np.float32)

    # Fix numerical errors, as values, that are actually 0,
    # might not be calculated as such
    tol = 1e-4 * (len_a + len_b + len_c)
    box[np.abs(box) < tol] = 0

    return box


def unitcell_from_vectors(box):
    """
    Get the unit cell lengths and angles from box vectors.

    This is the reverse operation of :func:`vectors_from_unitcell()`.

    Parameters
    ----------
    box : ndarray, shape=(3,3)
        The box vectors.

    Returns
    -------
    len_a, len_b, len_c : float
        The lengths of the three box/unit cell vectors *a*, *b* and *c*.
    alpha, beta, gamma : float
        The angles between the box vectors in radians.

    See Also
    --------
    vectors_from_unitcell : The reverse operation.
    """
    a = box[0]
    b = box[1]
    c = box[2]
    len_a = linalg.norm(a)
    len_b = linalg.norm(b)
    len_c = linalg.norm(c)
    alpha = np.arccos(np.dot(b, c) / (len_b * len_c))
    beta = np.arccos(np.dot(a, c) / (len_a * len_c))
    gamma = np.arccos(np.dot(a, b) / (len_a * len_b))
    return len_a, len_b, len_c, alpha, beta, gamma


def box_volume(box):
    """
    Get the volume of one ore multiple boxes.

    Parameters
    ----------
    box : ndarray, shape=(3,3) or shape=(m,3,3)
        One or multiple boxes to get the volume for.

    Returns
    -------
    volume : float or ndarray, shape=(m,)
        The volume(s) of the given box(es).
    """
    # Using the triple product
    return np.abs(linalg.det(box))


def repeat_box(atoms, amount=1):
    r"""
    Repeat the atoms in a box by duplicating and placing them in
    adjacent boxes.

    The output atom array (stack) contains the original atoms (central
    box) and duplicates of them in the given amount of adjacent boxes.
    The coordinates of the duplicate atoms are translated accordingly
    by the box coordinates.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms to be repeated.
        If `atoms` is a :class:`AtomArrayStack`, the atoms are repeated
        for each model, according to the box of each model.
    amount : int, optional
        The amount of boxes that are created in each direction of the
        central box.
        Hence, the total amount of boxes is
        :math:`(1 + 2 \cdot \text{amount}) ^ 3`.
        By default, one box is created in each direction, totalling in
        27 boxes.

    Returns
    -------
    repeated : AtomArray or AtomArrayStack
        The repeated atoms.
        Includes the original atoms (central box) in the beginning of
        the atom array (stack).
        If the input contains the ``sym_id`` annotation, the IDs are continued in the
        repeated atoms, i.e. they do not start at 0 again.
    indices : ndarray, dtype=int, shape=(n,3)
        Indices to the atoms in the original atom array (stack).
        Equal to
        ``numpy.tile(np.arange(atoms.array_length()), (1 + 2 * amount) ** 3)``.

    See Also
    --------
    repeat_box_coord : Variant that acts directly on coordinates.

    Examples
    --------

    >>> array = AtomArray(length=2)
    >>> array.coord = np.array([[1,5,3], [-1,2,5]], dtype=float)
    >>> array.box = np.array([[10,0,0], [0,10,0], [0,0,10]], dtype=float)
    >>> repeated, indices = repeat_box(array)
    >>> print(repeated.coord)
    [[  1.   5.   3.]
     [ -1.   2.   5.]
     [ -9.  -5.  -7.]
     [-11.  -8.  -5.]
     [ -9.  -5.   3.]
     [-11.  -8.   5.]
     [ -9.  -5.  13.]
     [-11.  -8.  15.]
     [ -9.   5.  -7.]
     [-11.   2.  -5.]
     [ -9.   5.   3.]
     [-11.   2.   5.]
     [ -9.   5.  13.]
     [-11.   2.  15.]
     [ -9.  15.  -7.]
     [-11.  12.  -5.]
     [ -9.  15.   3.]
     [-11.  12.   5.]
     [ -9.  15.  13.]
     [-11.  12.  15.]
     [  1.  -5.  -7.]
     [ -1.  -8.  -5.]
     [  1.  -5.   3.]
     [ -1.  -8.   5.]
     [  1.  -5.  13.]
     [ -1.  -8.  15.]
     [  1.   5.  -7.]
     [ -1.   2.  -5.]
     [  1.   5.  13.]
     [ -1.   2.  15.]
     [  1.  15.  -7.]
     [ -1.  12.  -5.]
     [  1.  15.   3.]
     [ -1.  12.   5.]
     [  1.  15.  13.]
     [ -1.  12.  15.]
     [ 11.  -5.  -7.]
     [  9.  -8.  -5.]
     [ 11.  -5.   3.]
     [  9.  -8.   5.]
     [ 11.  -5.  13.]
     [  9.  -8.  15.]
     [ 11.   5.  -7.]
     [  9.   2.  -5.]
     [ 11.   5.   3.]
     [  9.   2.   5.]
     [ 11.   5.  13.]
     [  9.   2.  15.]
     [ 11.  15.  -7.]
     [  9.  12.  -5.]
     [ 11.  15.   3.]
     [  9.  12.   5.]
     [ 11.  15.  13.]
     [  9.  12.  15.]]
    >>> print(indices)
    [0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0
     1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]

    The ``sym_id`` is continued in the repeated atoms.

    >>> array.set_annotation("sym_id", np.array([0, 0]))
    >>> repeated, indices = repeat_box(array)
    >>> print(repeated.sym_id)
    [ 0  0  1  1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9 10 10 11 11
     12 12 13 13 14 14 15 15 16 16 17 17 18 18 19 19 20 20 21 21 22 22 23 23
     24 24 25 25 26 26]
    """
    if atoms.box is None:
        raise BadStructureError("Structure has no box")

    repeat_coord, indices = repeat_box_coord(atoms.coord, atoms.box)
    # Unroll repeated coordinates for input to 'repeat()'
    if repeat_coord.ndim == 2:
        repeat_coord = repeat_coord.reshape(-1, atoms.array_length(), 3)
    else:  # ndim == 3
        repeat_coord = repeat_coord.reshape(
            atoms.stack_depth(), -1, atoms.array_length(), 3
        )
        repeat_coord = np.swapaxes(repeat_coord, 0, 1)

    repeated_atoms = repeat(atoms, repeat_coord)
    if "sym_id" in atoms.get_annotation_categories():
        max_sym_id = np.max(atoms.sym_id)
        # for the first repeat, (max_sym_id + 1) is added,
        # for the second repeat 2*(max_sym_id + 1) etc.
        repeated_atoms.sym_id += (max_sym_id + 1) * (
            np.arange(repeated_atoms.array_length()) // atoms.array_length()
        )
    return repeated_atoms, indices


def repeat_box_coord(coord, box, amount=1):
    r"""
    Similar to :func:`repeat_box()`, repeat the coordinates in a box by
    duplicating and placing them in adjacent boxes.

    Parameters
    ----------
    coord : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The coordinates to be repeated.
    box :  ndarray, dtype=float, shape=(3,3) or shape=(m,3,3)
        The reference box.
        If only one box is provided, i.e. the shape is *(3,3)*,
        the box is used for all models *m*, if the coordinates shape
        is *(m,n,3)*.
    amount : int, optional
        The amount of boxes that are created in each direction of the
        central box.
        Hence, the total amount of boxes is
        :math:`(1 + 2 \cdot \text{amount}) ^ 3`.
        By default, one box is created in each direction, totalling in
        27 boxes.

    Returns
    -------
    repeated : ndarray, dtype=float, shape=(p,3) or shape=(m,p,3)
        The repeated coordinates, with the same dimension as the input
        `coord`.
        Includes the original coordinates (central box) in the beginning
        of the array.
    indices : ndarray, dtype=int, shape=(p,3)
        Indices to the coordinates in the original array.
        Equal to
        ``numpy.tile(np.arange(coord.shape[-2]), (1 + 2 * amount) ** 3)``.
    """
    if not isinstance(amount, Integral):
        raise TypeError("The amount must be an integer")
    # List of numpy arrays for each box repeat
    coords_for_boxes = [coord]
    for i in range(-amount, amount + 1):
        for j in range(-amount, amount + 1):
            for k in range(-amount, amount + 1):
                # Omit the central box
                if i != 0 or j != 0 or k != 0:
                    temp_coord = coord.copy()
                    # Shift coordinates to adjacent box/unit cell
                    translation_vec = np.sum(
                        box * np.array([i, j, k])[:, np.newaxis], axis=-2
                    )
                    # 'newaxis' to perform same translation on all
                    # atoms for each model
                    temp_coord += translation_vec[..., np.newaxis, :]
                    coords_for_boxes.append(temp_coord)
    return (
        np.concatenate(coords_for_boxes, axis=-2),
        np.tile(np.arange(coord.shape[-2]), (1 + 2 * amount) ** 3),
    )


def move_inside_box(coord, box):
    r"""
    Move all coordinates into the given box, with the box vectors
    originating at *(0,0,0)*.

    Coordinates are outside the box, when they cannot be represented by
    a linear combination of the box vectors with scalar factors
    :math:`0 \le a_i \le 1`.
    In this case the affected coordinates are translated by the box
    vectors, so that they are inside the box.

    Parameters
    ----------
    coord : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The coordinates for one or multiple models.
    box : ndarray, dtype=float, shape=(3,3) or shape=(m,3,3)
        The box(es) for one or multiple models.
        When `coord` is given for multiple models, :attr:`box` must be
        given for multiple models as well.

    Returns
    -------
    moved_coord : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The moved coordinates.
        Has the same shape is the input `coord`.

    Examples
    --------

    >>> box = np.array([[10,0,0], [0,10,0], [0,0,10]], dtype=float)
    >>> inside_coord        = [ 1,  2,  3]
    >>> outside_coord       = [ 1, 22, 54]
    >>> other_outside_coord = [-4,  8,  6]
    >>> coord = np.stack([inside_coord, outside_coord, other_outside_coord])
    >>> print(coord)
    [[ 1  2  3]
     [ 1 22 54]
     [-4  8  6]]
    >>> moved_coord = move_inside_box(coord, box)
    >>> print(moved_coord.astype(int))
    [[1 2 3]
     [1 2 4]
     [6 8 6]]
    """
    fractions = coord_to_fraction(coord, box)
    fractions_rem = fractions % 1
    return fraction_to_coord(fractions_rem, box)


def remove_pbc(atoms, selection=None):
    """
    Remove segmentation caused by periodic boundary conditions from each
    molecule in the given structure.

    In this process the centroid of each molecule is moved into the
    dimensions of the box.
    To determine the molecules the structure is required to have an
    associated `BondList`.
    Otherwise segmentation removal is performed on a per-chain basis.

    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The potentially segmented structure.
        The :attr:`box` attribute must be set in the structure.
        An associated :attr:`bonds` attribute is recommended.
    selection : ndarray, dtype=bool, shape=(n,)
        Specifies which parts of `atoms` are sanitized, i.e the
        segmentation is removed.

    Returns
    -------
    sanitized_atoms : AtomArray or AtomArrayStack
        The input structure with removed segmentation over periodic
        boundaries.

    See Also
    --------
    remove_pbc_from_coord : Variant that acts directly on coordinates.

    Notes
    -----
    This function ensures that adjacent atoms in the input
    :class:`AtomArray`/:class:`AtomArrayStack` are spatially close to
    each other, i.e. their distance to each other is be smaller than the
    half box size.
    """
    # Avoid circular import
    from biotite.structure.geometry import centroid

    if atoms.box is None:
        raise BadStructureError("The 'box' attribute must be set in the structure")
    new_atoms = atoms.copy()

    if atoms.bonds is not None:
        molecule_masks = get_molecule_masks(atoms)
    else:
        molecule_masks = get_chain_masks(atoms, get_chain_starts(atoms))

    for mask in molecule_masks:
        if selection is not None:
            mask &= selection
        # Remove segmentation in molecule
        new_atoms.coord[..., mask, :] = remove_pbc_from_coord(
            new_atoms.coord[..., mask, :], atoms.box
        )
        # Put center of molecule into box
        center = centroid(new_atoms.coord[..., mask, :])[..., np.newaxis, :]
        center_in_box = move_inside_box(center, new_atoms.box)
        new_atoms.coord[..., mask, :] += center_in_box - center

    return new_atoms


def remove_pbc_from_coord(coord, box):
    """
    Remove segmentation caused by periodic boundary conditions from
    given coordinates.

    In this process the first coordinate is taken as origin and
    is moved inside the box.
    All other coordinates are assembled relative to the origin by using
    the displacement coordinates in adjacent array positions.

    Parameters
    ----------
    coord : ndarray, dtype=float, shape=(m,n,3) or shape=(n,3)
        The coordinates of the potentially segmented structure.
    box : ndarray, dtype=float, shape=(m,3,3) or shape=(3,3)
        The simulation box or unit cell that is used as periodic
        boundary.
        The amount of dimensions must fit the `coord` parameter.

    Returns
    -------
    sanitized_coord : ndarray, dtype=float, shape=(m,n,3) or shape=(n,3)
        The reassembled coordinates.

    See Also
    --------
    move_inside_box : The reverse operation.

    Notes
    -----
    This function solves a common problem from MD simulation output:
    When atoms move over the periodic boundary, they reappear on the
    other side of the box, segmenting the structure.
    This function reassembles the given coordinates, removing the
    segmentation.
    """

    # Import in function to avoid circular import
    from biotite.structure.geometry import index_displacement

    # Get the PBC-sanitized displacements of all coordinates
    # to the respective next coordinate
    index_pairs = np.stack(
        [np.arange(0, coord.shape[-2] - 1), np.arange(1, coord.shape[-2])], axis=1
    )
    neighbour_disp = index_displacement(coord, index_pairs, box=box, periodic=True)
    # Get the PBC-sanitized displacements of all but the first
    # coordinates to (0,0,0)
    absolute_disp = np.cumsum(neighbour_disp, axis=-2)
    # The first coordinate should be inside the box
    base_coord = move_inside_box(coord[..., 0:1, :], box)
    # The new coordinates are obtained by adding the displacements
    # to the new origin
    sanitized_coord = np.zeros(coord.shape, coord.dtype)
    sanitized_coord[..., 0:1, :] = base_coord
    sanitized_coord[..., 1:, :] = base_coord + absolute_disp
    return sanitized_coord


def coord_to_fraction(coord, box):
    """
    Transform coordinates to fractions of box vectors.

    Parameters
    ----------
    coord : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The coordinates for one or multiple models.
    box : ndarray, dtype=float, shape=(3,3) or shape=(m,3,3)
        The box(es) for one or multiple models.
        When `coord` is given for multiple models, :attr:`box` must be
        given for multiple models as well.

    Returns
    -------
    fraction : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The fractions of the box vectors.

    See Also
    --------
    fraction_to_coord : The reverse operation.

    Examples
    --------

    >>> box = np.array([[5,0,0], [0,5,0], [0,5,5]], dtype=float)
    >>> coord = np.array(
    ...     [[1,1,1], [10,0,0], [0,0,10], [-5,2,1]],
    ...     dtype=float
    ... )
    >>> print(coord)
    [[ 1.  1.  1.]
     [10.  0.  0.]
     [ 0.  0. 10.]
     [-5.  2.  1.]]
    >>> fractions = coord_to_fraction(coord, box)
    >>> print(fractions)
    [[ 0.2  0.0  0.2]
     [ 2.0  0.0  0.0]
     [ 0.0 -2.0  2.0]
     [-1.0  0.2  0.2]]
    """
    return np.matmul(coord, linalg.inv(box))


def fraction_to_coord(fraction, box):
    """
    Transform fractions of box vectors to coordinates.

    This is the reverse operation of :func:`coord_to_fraction()`.

    Parameters
    ----------
    fraction : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The fractions of the box vectors for one or multiple models.
    box : ndarray, dtype=float, shape=(3,3) or shape=(m,3,3)
        The box(es) for one or multiple models.
        When `coord` is given for multiple models, :attr:`box` must be
        given for multiple models as well.

    Returns
    -------
    coord : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The coordinates.

    See Also
    --------
    coord_to_fraction : The reverse operation.
    """
    return np.matmul(fraction, box)


def is_orthogonal(box):
    """
    Check, whether a box or multiple boxes is/are orthogonal.

    A box is orthogonal when the dot product of all box vectors with
    each other is 0.

    Parameters
    ----------
    box : ndarray, dtype=float, shape=(3,3) or shape=(m,3,3)
        A single box or multiple boxes.

    Returns
    -------
    is_orthgonal : bool or ndarray, shape=(m,), dtype=bool
        True, if the box vectors are orthogonal, false otherwise.

    Notes
    -----
    Due to possible numerical errors, this function also evaluates two
    vectors as orthogonal, when their dot product is not exactly zero,
    but it is within a small tolerance (:math:`10^{-6}`).
    """
    # Fix numerical errors, as values, that are actually 0,
    # might not be calculated as such
    tol = 1e-6
    return (
        (np.abs(vector_dot(box[..., 0, :], box[..., 1, :])) < tol)
        & (np.abs(vector_dot(box[..., 0, :], box[..., 2, :])) < tol)
        & (np.abs(vector_dot(box[..., 1, :], box[..., 2, :])) < tol)
    )


@functools.cache
def _get_transformation_data():
    with open(Path(__file__).parent / "spacegroups.json") as file:
        return json.load(file)
