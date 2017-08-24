# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module allows checking of atom arrays and atom array stacks for
errors in the structure.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack
from .filter import filter_backbone

__all__ = ["check_id_continuity", "check_bond_continuity"]


def check_id_continuity(array):
    """
    Check if the residue IDs are incremented by more than 1 or
    decremented.
    
    An increment by more than 1 is as strong clue for missung residues,
    a decrement means probably a start of a new chain.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    
    Returns
    -------
    discontinuity : ndarray(dtype=bool)
        True at the indices after a discontinuity.
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
    no bond at between those atoms, therefore the chain is discontinued.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    min_len, max_len : float, optional
        The interval in which the atom-atom distance is evaluated as
        bond.
    
    Returns
    -------
    discontinuity : ndarray(dtype=bool)
        True at the indices after a discontinuity.
    """
    backbone = array[filter_backbone(array)]
    diff = np.diff(backbone.coord, axis=0)
    sq_distance = np.sum(diff**2, axis=1)
    sq_min_len = min_len**2
    sq_max_len = max_len**2
    discontinuity = np.where( ((sq_distance < sq_min_len) | (sq_distance > sq_max_len)) )
    return discontinuity[0] + 1