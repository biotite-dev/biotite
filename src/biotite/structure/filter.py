# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility functions for creating filters on atom
arrays and atom array stacks.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["filter_solvent", "filter_monoatomic_ions", "filter_amino_acids",
           "filter_backbone", "filter_intersection",
           "filter_altloc"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack


_ext_aa_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                "MSE", "ASX", "GLX", "SEC", "UNK"]

_solvent_list = ["HOH","SOL"]


def filter_monoatomic_ions(array):
    """
    Filter all atoms of an atom array, that are monoatomic ions
    (e.g. sodium or chloride ions).
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        is a monoatomic ion.
    """
    # Exclusively in monoatomic ions,
    # the element name is equal to the residue name
    return (array.res_name == array.element)


def filter_solvent(array):
    """
    Filter all atoms of one array that are part of the solvent.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        belongs to the solvent.
    """
    return np.in1d(array.res_name, _solvent_list)


def filter_amino_acids(array):
    """
    Filter all atoms of one array that belong to amino acid residues.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray, dtype=bool
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
    filter : ndarray, dtype=bool
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
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        exists also in `intersect`.
    
    Examples
    --------
    
    Creating an atom array from atoms:
    
    >>> array1 = AtomArray(length=5)
    >>> array1.chain_id = np.array(["A","B","C","D","E"])
    >>> array2 = AtomArray(length=3)
    >>> array2.chain_id = np.array(["D","B","C"])
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


def filter_altloc(atoms, altlocs, selected_altlocs):
    """
    Filter all atoms having the desired altloc.
    
    Structure files (PDB, PDBx, MMTF) allow for duplicate atom records,
    in case a residue is found in multiple alternative locations
    (*altloc*).
    This function is used to filter the atoms with the desired *altloc*
    at this position with other *altlocs* are removed.    
    
    The function will be rarely used by the end user, since this kind
    of filtering is automatically performed, when the structure is
    loaded from a file.
    In the final atom array (stack) duplicate atoms are not allowed.
    
    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The unfiltered atom array to be filtered.
    altlocs : array-like, shape=(n,)
        An array containing the alternative location codes for each
        atom in the unfiltered atom array.
        Can contain '.', '?', ' ', '' or a letter at each position.
    selected_altlocs : iterable object of tuple (str, int, str)
        Each tuple consists of the following elements:

            - A chain ID, specifying the residue
            - A residue ID, specifying the residue
            - The desired *altoc* ID for the specified residue

        For each of the given residues only those atoms of `atoms` are
        filtered where the *altloc* ID matches the respective *altloc*
        ID in `altlocs`.
        By default the location with the *altloc* ID "A" is used.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        The combined inscode and altloc filters.
    """
    # Default: Filter all atoms
    # with altloc code ".", "?", "A", " " or empty string
    altloc_filter = np.in1d(altlocs, [".","?","A"," ",""])
    
    for loc in selected_altlocs:
        chain, residue, altoc = loc
        residue_filter = (
            (atoms.chain_id == chain) &
            (atoms.res_id == residue)
        )
        # Reset (set to False) filter for specified residue
        altloc_filter &= ~residue_filter
        # Choose only atoms of residue with altloc code
        altloc_filter |= residue_filter & (altlocs == altloc)
    
    return altloc_filter


