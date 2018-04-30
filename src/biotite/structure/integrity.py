# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module allows checking of atom arrays and atom array stacks for
errors in the structure.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["check_id_continuity", "check_bond_continuity",
           "check_duplicate_atoms"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack
from .filter import filter_backbone


def check_id_continuity(array):
    """
    Check if the residue IDs are incremented by more than 1 or
    decremented, from one atom to the next one.
    
    An increment by more than 1 is as strong clue for missung residues,
    a decrement means probably a start of a new chain.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    
    Returns
    -------
    discontinuity : ndarray, dtype=bool
        Contains the indices of atoms after a discontinuity
    """
    ids = array.res_id
    diff = np.diff(ids)
    discontinuity = np.where( ((diff != 0) & (diff != 1)) )
    return discontinuity[0] + 1


def check_bond_continuity(array, min_len=1.2, max_len=1.8):
    """
    Check if the peptide backbone atoms ("N","CA","C") have a
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
    discontinuity : ndarray, dtype=bool
         Contains the indices of atoms after a discontinuity.
    """
    backbone_mask = filter_backbone(array)
    backbone = array[backbone_mask]
    diff = np.diff(backbone.coord, axis=0)
    sq_distance = np.sum(diff**2, axis=1)
    sq_min_len = min_len**2
    sq_max_len = max_len**2
    discon_mask = (sq_distance < sq_min_len) | (sq_distance > sq_max_len)
    # discon_mask is shortened by 1 due to np.diff
    # -> Lenthening by 1 to fit backbone
    # and shifting by one to get position after discontinuity
    discon_mask = np.concatenate(([False], discon_mask))
    # discon_mask is true for discontinuity in backbone
    # map it back to get discontinuity for original atom array
    discon_mask_full = np.zeros(array.array_length(), dtype=bool)
    discon_mask_full[backbone_mask] = discon_mask
    discontinuity = np.where(discon_mask_full)
    return discontinuity[0]


def check_duplicate_atoms(array):
    """
    Check if a structure contains duplicate atoms, i.e. two atoms in a
    structure have the same annotations (coordinates may be different).
    
    Duplicate atoms may appear, when a structure has occupancy for an
    atom at two or more positions or when the *insertion code* or
    *altloc* are improperly read.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    
    Returns
    -------
    duplicate : ndarray, dtype=bool
        Contains the indices of duplicate atoms. The first occurence of
        an atom is not counted as duplicate.
    """
    duplicates = []
    annots = [array.get_annotation(category) for category
              in array.get_annotation_categories()]
    for i in range(1, array.array_length()):
        # Start with assumption that all atoms in the array
        # until index i are duplicates of the atom at index i
        is_dublicate = np.full(i, True, dtype=bool)
        for annot in annots:
            # For each annotation array filter out the atoms until
            # index i that have an unequal annotation
            # to the atom at index i 
            is_dublicate &= (annot[:i] == annot[i])
        # After checking all annotation arrays,
        # if there still is any duplicate to the atom at index i,
        # add i the the list of duplicate atom indices
        if is_dublicate.any():
            duplicates.append(i)
    return np.array(duplicates)

