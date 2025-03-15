# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows checking of atom arrays and atom array stacks for
errors in the structure.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann, Daniel Bauer"
__all__ = [
    "check_atom_id_continuity",
    "check_res_id_continuity",
    "check_backbone_continuity",
    "check_duplicate_atoms",
    "check_linear_continuity",
]

import numpy as np
from biotite.structure.box import coord_to_fraction
from biotite.structure.filter import (
    filter_linear_bond_continuity,
    filter_peptide_backbone,
    filter_phosphate_backbone,
)


def _check_continuity(array):
    diff = np.diff(array)
    discontinuity = np.where(((diff != 0) & (diff != 1)))
    return discontinuity[0] + 1


def check_atom_id_continuity(array):
    """
    Check if the atom IDs are incremented by more than 1 or
    decremented, from one atom to the next one.

    An increment by more than 1 is as strong clue for missing atoms.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.

    Returns
    -------
    discontinuity : ndarray, dtype=int
        Contains the indices of atoms after a discontinuity.
    """
    ids = array.atom_id
    return _check_continuity(ids)


def check_res_id_continuity(array):
    """
    Check if the residue IDs are incremented by more than 1 or
    decremented, from one atom to the next one.

    An increment by more than 1 is as strong clue for missing residues,
    a decrement means probably a start of a new chain.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.

    Returns
    -------
    discontinuity : ndarray, dtype=int
        Contains the indices of atoms after a discontinuity.
    """
    ids = array.res_id
    return _check_continuity(ids)


def check_linear_continuity(array, min_len=1.2, max_len=1.8):
    """
    Check linear (consecutive) bond continuity of atoms in atom array.

    Parameters
    ----------
    array : AtomArray
        Arbitrary structure.
    min_len : float, optional
        Minimum bond length.
    max_len : float, optional
        Maximum bond length.

    Returns
    -------
    discontinuity : ndarray, dtype=int
        Indices of `array` corresponding to atoms where the bond
        with the preceding atom is beyond the provided bounds.

    See Also
    --------
    filter_linear_bond_continuity : A function to filter for atoms preserving the continuity (used here).
    BondList : A class that doesn't depend on the atoms' order to identify bonds.
    """
    con_mask = filter_linear_bond_continuity(array, min_len, max_len)
    # The continuity mask `con_mask` points to atoms for which the next atom is continuous.
    # We invert this mask and shift-extend by one from the left.
    # The resulting discontinuity mask points to atoms having the preceding atom exceeding
    # the bond length requirements.
    discon_mask = np.insert(~con_mask[:-1], 0, False)
    return np.where(discon_mask)[0]


def check_backbone_continuity(array, min_len=1.2, max_len=1.8):
    """
    Check if the (peptide or phosphate) backbone atoms have
    non-reasonable distance to the next atom.

    A large or very small distance is a very strong clue, that there is
    no bond between those atoms, therefore the chain is discontinued.

    Parameters
    ----------
    array : AtomArray
        The array to be checked.
    min_len, max_len : float, optional
        The interval in which the atom-atom distance is evaluated as
        bond.

    Returns
    -------
    discontinuity : ndarray, dtype=int
         Contains the indices of atoms after a discontinuity.

    See Also
    --------
    filter_linear_bond_continuity : A function to filter for atoms preserving the continuity.
    filter_peptide_backbone : A function to filter for peptide backbone atoms.
    filter_phosphate_backbone : A function to filter for phosphate backbone atoms.
    """
    backbone_mask = filter_peptide_backbone(array) | filter_phosphate_backbone(array)
    con_mask = filter_linear_bond_continuity(array[backbone_mask], min_len, max_len)

    # See the comments for `check_linear_continuity()`
    discon_mask = np.insert(~con_mask[:-1], 0, False)
    discon_mask_full = np.full_like(backbone_mask, False)
    discon_mask_full[backbone_mask] = discon_mask

    return np.where(discon_mask_full)[0]


def check_duplicate_atoms(array):
    """
    Check if a structure contains duplicate atoms, i.e. two atoms in a
    structure have the same annotations (coordinates may be different).

    Duplicate atoms may appear, when a structure has occupancy for an
    atom at two or more positions or when the *altloc* positions are
    improperly read.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.

    Returns
    -------
    duplicate : ndarray, dtype=int
        Contains the indices of duplicate atoms.
        The first occurence of an atom is not counted as duplicate.
    """
    duplicates = []
    annots = [
        array.get_annotation(category) for category in array.get_annotation_categories()
    ]
    for i in range(1, array.array_length()):
        # Start with assumption that all atoms in the array
        # until index i are duplicates of the atom at index i
        is_duplicate = np.full(i, True, dtype=bool)
        for annot in annots:
            # For each annotation array filter out the atoms until
            # index i that have an unequal annotation
            # to the atom at index i
            is_duplicate &= annot[:i] == annot[i]
        # After checking all annotation arrays,
        # if there still is any duplicate to the atom at index i,
        # add i the the list of duplicate atom indices
        if is_duplicate.any():
            duplicates.append(i)
    return np.array(duplicates)


def check_in_box(array):
    r"""
    Check if a structure contains atoms whose position is outside the
    box.

    Coordinates are outside the box, when they cannot be represented by
    a linear combination of the box vectors with scalar factors
    :math:`0 \le a_i \le 1`.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.

    Returns
    -------
    outside : ndarray, dtype=int
        Contains the indices of atoms outside the atom array's box.
    """
    if array.box is None:
        raise TypeError("Structure has no box")
    box = array.box
    fractions = coord_to_fraction(array, box)
    return np.where(((fractions >= 0) & (fractions < 1)).all(axis=-1))[0]
