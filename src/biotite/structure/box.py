# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Functions related to working with the simulation box or unit cell
of a structure 
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["vectors_from_unitcell", "unitcell_from_vectors", "box_volume",
           "repeat_box", "repeat_box_coord", "move_inside_box",
           "remove_pbc", "remove_pbc_from_coord",
           "coord_to_fraction", "fraction_to_coord", "is_orthogonal"]

from collections.abc import Iterable
from numbers import Integral
import numpy as np
import numpy.linalg as linalg
from .util import vector_dot
from .residues import get_residue_starts
from .atoms import AtomArray, AtomArrayStack
from .error import BadStructureError


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
    alpha, beta, gamma:
        The angles between the box vectors in radians.
        *alpha* is the angle between *b* and *c*,
        *beta* between *a* and *c*, *gamma* between *a* and *b*
    
    Returns
    -------
    box : ndarray, dtype=float, shape=(3,3)
        The three box vectors.
        The vector components are in the last dimension.
        The value can be directly used as :attr:`box` attribute in an
        atom array.
    
    See also
    --------
    unitcell_from_vectors
    """
    a_x = len_a
    b_x = len_b * np.cos(gamma)
    b_y = len_b * np.sin(gamma)
    c_x = len_c * np.cos(beta)
    c_y = len_c * (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    c_z = np.sqrt(len_c*len_c - c_x*c_x - c_y*c_y)
    box = np.array([
        [a_x,   0,   0],
        [b_x, b_y,   0],
        [c_x, c_y, c_z]
    ], dtype=np.float32)
    
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
        The box vectors
    
    Returns
    -------
    len_a, len_b, len_c : float
        The lengths of the three box/unit cell vectors *a*, *b* and *c*.
    alpha, beta, gamma : float
        The angles between the box vectors in radians.

    See also
    --------
    vectors_from_unitcell
    """
    a = box[0]
    b = box[1]
    c = box[2]
    len_a = linalg.norm(a)
    len_b = linalg.norm(b)
    len_c = linalg.norm(c)
    alpha = np.arccos(np.dot(b, c) / (len_b * len_c))
    beta  = np.arccos(np.dot(a, c) / (len_a * len_c))
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
    indices : ndarray, dtype=int, shape=(n,3)
        Indices to the atoms in the original atom array (stack).
        Equal to
        ``numpy.tile(np.arange(atoms.array_length()), (1 + 2 * amount) ** 3)``.
    
    See also
    --------
    repeat_box_coord

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
    """
    if atoms.box is None:
        raise TypeError("Structure has no box")
    if not isinstance(amount, Integral):
        raise TypeError("The amount must be an integer")
    box_count = (1 + 2 * amount) ** 3
    if isinstance(atoms, AtomArray):
        repeat_atoms = AtomArray(
            box_count * atoms.array_length()
        )
    elif isinstance(atoms, AtomArrayStack):
        repeat_atoms = AtomArrayStack(
            atoms.stack_depth(), box_count * atoms.array_length()
        )
    else:
        raise TypeError("An atom array or stack is required")
    
    repeat_atoms.box = atoms.box.copy()
    if atoms.bonds is not None:
        repeat_bonds = atoms.bonds.copy()
        # Repeat the bonds list 'box_count' times
        for i in range(box_count-1):
            repeat_bonds += repeat_bonds
        repeat_atoms.bonds = atoms.repeat_bonds
    for category in atoms.get_annotation_categories():
        annot_array = atoms.get_annotation(category)
        # The atoms have the same annotation in all boxes
        repeat_atoms.set_annotation(category, np.tile(annot_array, box_count))
    
    repeat_coord, indices = repeat_box_coord(atoms.coord, atoms.box)
    repeat_atoms.coord = repeat_coord
    return repeat_atoms, indices


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
    repeated : ndarray, dtype=float, shape=(n,3) or shape=(m,n,3)
        The repeated coordinates, with the same shape as the input
        `coord`.
        Includes the original coordinates (central box) in the beginning
        of the array.
    indices : ndarray, dtype=int, shape=(n,3)
        Indices to the coordiantes in the original array.
        Equal to
        ``numpy.tile(np.arange(coord.shape[-2]), (1 + 2 * amount) ** 3)``.
    """
    if not isinstance(amount, Integral):
        raise TypeError("The amount must be an integer")
    # List of numpy arrays for each box repeat
    coords_for_boxes = [coord]
    for i in range(-amount, amount+1):
        for j in range(-amount, amount+1):
            for k in range(-amount, amount+1):
                # Omit the central box
                if i != 0 or j != 0 or k != 0:
                    temp_coord = coord.copy()
                    # Shift coordinates to adjacent box/unit cell
                    translation_vec = np.sum(
                        box * np.array([i,j,k])[:, np.newaxis],
                        axis=-2
                    )
                    # 'newaxis' to perform same translation on all
                    # atoms for each model
                    temp_coord += translation_vec[..., np.newaxis, :]
                    coords_for_boxes.append(temp_coord)
    return (
        np.concatenate(coords_for_boxes, axis=-2),
        np.tile(np.arange(coord.shape[-2]), (1 + 2 * amount) ** 3)
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
    Remove segmentation caused by periodic boundary conditions from a
    given structure.

    In this process the first atom (of the selection) is taken as origin
    and is moved inside the box.
    All other coordinates are assembled relative to the origin.
    
    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The potentially segmented structure.
        The :attr:`box` attribute must be set in the structure.
    selection : str or (iterable object of) ndarray, dtype=bool, shape=(n,), optional
        Specifies which part(s) of structure are sanitized, i.e the
        segmentation is removed.
        If a string is given, the value is interpreted as the chain ID
        to be selected.
        If a boolean mask is given, the corresponding atoms are
        selected. 
        If multiple boolean masks are given, each selection
        is treated as separate assembly process, independent of all
        other selections.
        Consequently, giving multiple boolean masks has the same result
        as calling the functions multiple times with each mask
        separately.
        An atom must not be selected more than one time.

    
    Returns
    -------
    sanitized_atoms : AtomArray or AtomArrayStack
        The input structure with removed periodic boundary conditions.
    
    See also
    --------
    remove_pbc_from_coord

    Notes
    -----
    It is not recommended to select regions of the
    structure with distances from one atom to the next atom that are
    larger than half of the box size
    (e.g. the solvent, chain transitions).
    In this case, multiple selections should be given, with a single
    molecule selected in each selection.

    Internally the function uses :func:`remove_pbc_from_coord()`.
    """
    if atoms.box is None:
        raise BadStructureError(
            "The 'box' attribute must be set in the structure"
        )
    new_atoms = atoms.copy()
    
    if selection is None:
        new_atoms.coord = remove_pbc_from_coord(
            atoms.coord, atoms.box
        )

    elif isinstance(selection, str):
        # Chain ID
        selection = (atoms.chain_id == selection)
        new_atoms.coord[..., selection, :] = remove_pbc_from_coord(
            atoms.coord[..., selection, :], atoms.box
        )
    
    elif isinstance(selection, np.ndarray) and selection.ndim == 1:
        # Single boolean mask
        new_atoms.coord[..., selection, :] = remove_pbc_from_coord(
            atoms.coord[..., selection, :], atoms.box
        )
    
    elif isinstance(selection, Iterable):
        # Iterable of boolean masks
        selections = np.stack(list(selection))
        # Test whether an atom was selected multiple times
        sel_count = np.count_nonzero(selections, axis=0)
        if (sel_count > 1).any():
            first_pos = np.where((sel_count > 1))[0][0]
            raise ValueError(
                f"Atom at index {first_pos} was selected "
                f"{sel_count[first_pos]} times"
            )
        for selection in selections:
            new_atoms.coord[..., selection, :] = remove_pbc_from_coord(
                atoms.coord[..., selection, :], atoms.box
            )

    return new_atoms


def remove_pbc_from_coord(coord, box):
    """
    Remove segmentation caused by periodic boundary conditions from
    given coordinates.

    In this process the first coordinate is taken as origin and
    is moved inside the box.
    All other coordinates are assembled relative to the origin by using
    the displacement coordinates in adjacent array positions.
    Basically, this function performs the reverse action of
    :func:`move_inside_box()`.
    
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
    
    See also
    --------
    remove_pbc_from_coord
    move_inside_box

    Notes
    -----
    This function solves a common problem from MD simulation output:
    When atoms move over the periodic boundary, they reappear on the
    other side of the box, segmenting the structure.
    This function reassembles the given coordinates, removing the
    segmentation.
    """

    # Import in function to avoid circular import
    from .geometry import index_displacement
    # Get the PBC-sanitized displacements of all coordinates
    # to the respective next coordinate
    index_pairs = np.stack(
        [
            np.arange(0, coord.shape[-2] - 1), 
            np.arange(1, coord.shape[-2]    )
        ],
        axis=1
    )
    neighbour_disp = index_displacement(
        coord, index_pairs, box=box, periodic=True
    )
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
    
    See also
    --------
    fraction_to_coord

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
    
    See also
    --------
    coord_to_fraction
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
        True, if the box vectors are orthogonal, false otherwise
    
    Notes
    -----
    Due to possible numerical errors, this function also evaluates two
    vectors as orthogonal, when their dot product is not exactly zero,
    but it is within a small tolerance (:math:`10^{-6}`).
    """
    # Fix numerical errors, as values, that are actually 0,
    # might not be calculated as such
    tol = 1e-6
    return (np.abs(vector_dot(box[..., 0, :], box[..., 1, :])) < tol) & \
           (np.abs(vector_dot(box[..., 0, :], box[..., 2, :])) < tol) & \
           (np.abs(vector_dot(box[..., 1, :], box[..., 2, :])) < tol)