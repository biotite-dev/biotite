# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility functions for creating filters on atom
arrays and atom array stacks.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["filter_solvent", "filter_monoatomic_ions", "filter_nucleotides",
           "filter_amino_acids", "filter_backbone", "filter_intersection",
           "filter_first_altloc", "filter_highest_occupancy_altloc"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack
from .residues import get_residue_starts


_ext_aa_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                "MSE", "ASX", "GLX", "SEC", "UNK"]

_ext_nucleotide_list = ["DT", "T", "DA", "A", "DG", "G", "DC", "C", "DU", "U",
                        "H2U", "I", "JMH", "5MC", "4OC", "U8U", "4SU", "PSU",
                        "4AC"]

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


def filter_nucleotides(array):
    """
    Filter all atoms of one array that belong to nucleotides.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        belongs to a nucleotide.
    """
    return (
        np.in1d(array.res_name, _ext_nucleotide_list) 
        & (array.res_id != -1)
    )


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


def filter_first_altloc(atoms, altloc_ids):
    """
    Filter all atoms, that have the first *altloc* ID appearing in a
    residue.
    
    Structure files (PDB, PDBx, MMTF) allow for duplicate atom records,
    in case a residue is found in multiple alternate locations
    (*altloc*).
    This function is used to remove such duplicate atoms by choosing a
    single *altloc ID* for an atom with other *altlocs* being removed.    
    
    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The unfiltered structure to be filtered.
    altloc_ids : ndarray, shape=(n,), dtype='U1'
        An array containing the alternate location IDs for each
        atom in `atoms`.
        Can contain `'.'`, `'?'`, `' '`, `''` or a letter at each
        position.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        For each residue, this array is True in the following cases:

            - The atom has no altloc ID (`'.'`, `'?'`, `' '`, `''`).
            - The atom has the same altloc ID (e.g. `'A'`, `'B'`, etc.)
              as the first atom in the residue that has an altloc ID.
    
    Notes
    -----
    The function will be rarely used by the end user, since this kind
    of filtering is usually automatically performed, when the structure
    is loaded from a file.
    The exception are structures that were read with `altloc` set to
    `True`.

    Examples
    --------

    >>> atoms = array([
    ...     Atom(coord=[1, 2, 3], res_id=1, atom_name="CA"),
    ...     Atom(coord=[4, 5, 6], res_id=1, atom_name="CB"),
    ...     Atom(coord=[6, 5, 4], res_id=1, atom_name="CB")
    ... ])
    >>> altloc_ids = np.array([".", "A", "B"])
    >>> filtered = atoms[filter_first_altloc(atoms, altloc_ids)]
    >>> print(filtered)
                1      CA               1.000    2.000    3.000
                1      CB               4.000    5.000    6.000
    """
    # Filter all atoms without altloc code
    altloc_filter = np.in1d(altloc_ids, [".", "?", " ", ""])
    
    # And filter all atoms for each residue with the first altloc ID
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        letter_altloc_ids = [l for l in altloc_ids[start:stop] if l.isalpha()]
        if len(letter_altloc_ids) > 0:
            first_id = letter_altloc_ids[0]
            altloc_filter[start:stop] |= (altloc_ids[start:stop] == first_id)
        else:
            # No altloc ID in this residue -> Nothing to do
            pass
    
    return altloc_filter


def filter_highest_occupancy_altloc(atoms, altloc_ids, occupancies):
    """
    For each residue, filter all atoms, that have the *altloc* ID
    with the highest occupancy for this residue.
    
    Structure files (PDB, PDBx, MMTF) allow for duplicate atom records,
    in case a residue is found in multiple alternate locations
    (*altloc*).
    This function is used to remove such duplicate atoms by choosing a
    single *altloc ID* for an atom with other *altlocs* being removed.    
    
    Parameters
    ----------
    atoms : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The unfiltered structure to be filtered.
    altloc_ids : ndarray, shape=(n,), dtype='U1'
        An array containing the alternate location IDs for each
        atom in `atoms`.
        Can contain `'.'`, `'?'`, `' '`, `''` or a letter at each
        position.
    occupancies : ndarray, shape=(n,), dtype=float
        An array containing the occupancy values for each atom in
        `atoms`.
    
    Returns
    -------
    filter : ndarray, dtype=bool
        For each residue, this array is True in the following cases:

            - The atom has no altloc ID
              (``'.'``, ``'?'``, ``' '``, ``''``).
            - The atom has the altloc ID (e.g. ``'A'``, ``'B'``, etc.),
              of which the corresponding occupancy values are highest
              for the **entire** residue.
    
    Notes
    -----
    The function will be rarely used by the end user, since this kind
    of filtering is usually automatically performed, when the structure
    is loaded from a file.
    The exception are structures that were read with ``altloc`` set to
    ``True``.

    Examples
    --------

    >>> atoms = array([
    ...     Atom(coord=[1, 2, 3], res_id=1, atom_name="CA"),
    ...     Atom(coord=[4, 5, 6], res_id=1, atom_name="CB"),
    ...     Atom(coord=[6, 5, 4], res_id=1, atom_name="CB")
    ... ])
    >>> altloc_ids = np.array([".", "A", "B"])
    >>> occupancies = np.array([1.0, 0.1, 0.9])
    >>> filtered = atoms[filter_highest_occupancy_altloc(
    ...     atoms, altloc_ids, occupancies
    ... )]
    >>> print(filtered)
                1      CA               1.000    2.000    3.000
                1      CB               6.000    5.000    4.000
    """
    # Filter all atoms without altloc code
    altloc_filter = np.in1d(altloc_ids, [".", "?", " ", ""])
    
    # And filter all atoms for each residue with the highest sum of
    # occupancies
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        occupancies_in_res = occupancies[start:stop]
        altloc_ids_in_res = altloc_ids[start:stop]
        
        letter_altloc_ids = [l for l in altloc_ids_in_res if l.isalpha()]
        
        if len(letter_altloc_ids) > 0:
            highest = -1.0
            highest_id = None
            for id in set(letter_altloc_ids):
                occupancy_sum = np.sum(
                    occupancies_in_res[altloc_ids_in_res == id]
                )
                if occupancy_sum > highest:
                    highest = occupancy_sum
                    highest_id = id
            altloc_filter[start:stop] |= (altloc_ids[start:stop] == highest_id)
        else:
            # No altloc ID in this residue -> Nothing to do
            pass
    
    return altloc_filter
