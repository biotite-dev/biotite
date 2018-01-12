# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

"""
This module provides utility functions for creating filters on atom
arrays and atom array stacks.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack

__all__ = ["filter_solvent", "filter_amino_acids", "filter_backbone",
           "filter_intersection"]


_ext_aa_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                "MSE", "ASX", "GLX", "SEC", "UNK"]

_solvent_list = ["HOH","SOL"]


def filter_solvent(array):
    """
    Filter all atoms of one array that are part of the solvent.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray(dtype=bool)
        This array is `True` for all indices in `array`, where the atom
        belongs to the solvent.
    """
    return ( np.in1d(array.res_name, _solvent_list) & (array.hetero == True) )


def filter_amino_acids(array):
    """
    Filter all atoms of one array that belong to amino acid residues.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray(dtype=bool)
        This array is `True` for all indices in `array`, where the atom
        belongs to an amino acid residue.
    """
    return ( np.in1d(array.res_name, _ext_aa_list) & (array.res_id != -1) )


def filter_backbone(array):
    """
    Filter all peptide backbone atoms of one array.
    
    This includes the "N", "CA" and "C" atoms of amino acids.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray(dtype=bool)
        This array is `True` for all indices in `array`, where the atom
        as an backbone atom.
    """
    return ( ((array.atom_name == "N") |
              (array.atom_name == "CA") |
              (array.atom_name == "C")) &
              filter_amino_acids(array) )

    
def filter_intersection(array, intersect):
    """
    Filter all atoms of one array that exist also in another array.
    
    An atom is defined as existent in the second array, if there is an
    atom in the second array that has the same annotation values in all
    categories that exists in both arrays.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    intersect : AtomArray
        Atoms in `array` that also exists in `intersect` are filtered.
    
    Returns
    -------
    filter : ndarray(dtype=bool)
        This array is `True` for all indices in `array`, where the atom
        exists also in `intersect`.
    
    Examples
    --------
    
    Creating an atom array from atoms:
    
        >>> array1 = AtomArray(length=5)
        >>> array1.chain_id = np.array("A","B","C","D","E")
        >>> array2 = AtomArray(length=5)
        >>> array2.chain_id = np.array("D","B","C")
        >>> array1 = array1[filter_intersection(array1, array2)]
        >>> print(array1.chain_id)
        ['B' 'C' 'D']
    
    """
    filter = np.full(array.array_length(), True, dtype=bool)
    intersect_categories = intersect.get_annotation_categories()
    # Check atom equality only for categories,
    # which exist in both arrays
    categories = [category for category in array.get_annotation_categories()
                  if category in intersect_categories]
    for i in range(array.array_length()):
        subfilter = np.full(intersect.array_length(), True, dtype=bool)
        for category in categories:
            subfilter &= (intersect.get_annotation(category)
                          == array.get_annotation(category)[i])
        filter[i] = subfilter.any()
    return filter


def filter_inscode_and_altloc(array, inscode=[], altloc=[],
                              altloc_array=None, inscode_array=None):
    """
    Filter all atoms having the desired altloc or inscode.
    
    Structure files (PDB, PDBx, MMTF) allow for duplicate atom records,
    in case a residue is found in multiple alternative locations
    (altloc) or different residues are inserted at a specific location
    (inscode). This function is used to filter the desired altlocs and
    inscodes, atoms at this position with other altlocs or inscodes are
    removed.    
    
    The function will be merely used by the end user, since this kind
    of filtering is automatically performed, when the structure is
    loaded from a file. In the final atom array (stack) duplicate atoms
    are not allowed.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    intersect : AtomArray
        Atoms in `array` that also exists in `intersect` are filtered.
    
    Returns
    -------
    filter : ndarray(dtype=bool)
        This array is `True` for all indices in `array`, where the atom
        exists also in `intersect`.
    """
    if inscode_array is None:
        # In case no insertion code column is existent
        inscode_filter = np.full(array.array_length(), True)
    else:
        # Default: Filter all atoms
        # with insertion code ".", "?" or " "
        inscode_filter = np.in1d(inscode_array, [".","?"])
        # Now correct filter for every given insertion code
        for code in inscode:
            residue = code[0]
            insertion = code[1]
            residue_filter = (array.res_id == residue)
            # Reset (to False) filter for given res_id
            inscode_filter &= ~residue_filter
            # Choose atoms of res_id with insertion code
            inscode_filter |= residue_filter & (inscode_array == insertion)
    # Same with altlocs
    if altloc_array is None:
        altloc_filter = np.full(array.array_length(), True)
    else:
        altloc_array = model_dict["label_alt_id"]
        altloc_filter = np.in1d(altloc_array, [".","?","A"," "])
        for loc in altloc:
            residue = loc[0]
            altloc = loc[1]
            residue_filter = (array.res_id == residue)
            altloc_filter &= ~residue_filter
            altloc_filter |= residue_filter & (altloc_array == altloc)
    # return combined filters
    return inscode_filter & altloc_filter


