# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains functionalities for repairing malformed structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["renumber_atom_ids", "renumber_res_ids"]

import numpy as np


def renumber_atom_ids(array, start=None):
    """
    Renumber the atom IDs of the given array.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    start : int, optional
        The starting index for renumbering.
        The first ID in the array is taken by default.

    Returns
    -------
    array : AtomArray or AtomArrayStack
        The renumbered array.
    """
    if "atom_id" not in array.get_annotation_categories():
        raise ValueError("The atom array must have the 'atom_id' annotation")
    if start is None:
        start = array.atom_id[0]
    array = array.copy()
    array.atom_id = np.arange(start, array.shape[-1]+1)
    return array


def renumber_res_ids(array, start=None):
    """
    Renumber the residue IDs of the given array, so that are continuous.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be checked.
    start : int, optional
        The starting index for renumbering.
        The first ID in the array is taken by default.

    Returns
    -------
    array : AtomArray or AtomArrayStack
        The renumbered array.
    """
    if start is None:
        start = array.res_id[0]
    diff = np.diff(array.res_id)
    diff[diff != 0] = 1
    new_res_ids =  np.concatenate(([start], diff)).cumsum()
    array = array.copy()
    array.res_id = new_res_ids
    return array
