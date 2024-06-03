# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module contains functionalities for repairing malformed structures.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann, Daniel Bauer"
__all__ = ["renumber_atom_ids", "renumber_res_ids",
           "create_continuous_res_ids", "infer_elements", "create_atom_names"]

from collections import Counter
import warnings
import numpy as np
from .atoms import AtomArray, AtomArrayStack
from .residues import get_residue_starts
from .chains import get_chain_starts


def renumber_atom_ids(array, start=None):
    """
    Renumber the atom IDs of the given array.

    DEPRECATED.

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
    warnings.warn(
      "'renumber_atom_ids()' is deprecated",
        DeprecationWarning
    )
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

    DEPRECATED: Use :func:`create_continuous_res_ids()`instead.

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
    warnings.warn(
      "'renumber_res_ids()' is deprecated, use 'create_continuous_res_ids()'",
        DeprecationWarning
    )
    if start is None:
        start = array.res_id[0]
    diff = np.diff(array.res_id)
    diff[diff != 0] = 1
    new_res_ids =  np.concatenate(([start], diff)).cumsum()
    array = array.copy()
    array.res_id = new_res_ids
    return array


def create_continuous_res_ids(atoms, restart_each_chain=True):
    """
    Create an array of continuous residue IDs for a given structure.

    This means that residue IDs are incremented by 1 for each residue.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The atoms for which the continuous residue IDs should be created.
    restart_each_chain : bool, optional
        If true, the residue IDs are reset to 1 for each chain.

    Returns
    -------
    res_ids : ndarray, dtype=int
        The continuous residue IDs.

    Examples
    --------

    >>> # Remove a residue to make the residue IDs discontinuous
    >>> atom_array = atom_array[atom_array.res_id != 5]
    >>> res_ids, _ = get_residues(atom_array)
    >>> print(res_ids)
    [ 1  2  3  4  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    >>> atom_array.res_id = create_continuous_res_ids(atom_array)
    >>> res_ids, _ = get_residues(atom_array)
    >>> print(res_ids)
    [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]

    """
    res_ids_diff = np.zeros(atoms.array_length(), dtype=int)
    res_starts = get_residue_starts(atoms)
    res_ids_diff[res_starts] = 1
    res_ids = np.cumsum(res_ids_diff)

    if restart_each_chain:
        chain_starts = get_chain_starts(atoms)
        for start in chain_starts:
            res_ids[start:] -= res_ids[start] - 1

    return res_ids


def infer_elements(atoms):
    """
    Infer the elements of atoms based on their atom name.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or array-like of str
        The atoms for which the elements should be inferred.
        Alternatively the atom names can be passed directly.

    Returns
    -------
    elements : ndarray, dtype=str
        The inferred elements.

    See Also
    --------
    create_atoms_names : The opposite of this function

    Examples
    --------

    >>> print(infer_elements(atom_array)[:10])
    ['N' 'C' 'C' 'O' 'C' 'C' 'O' 'N' 'H' 'H']
    >>> print(infer_elements(["CA", "C", "C1", "OD1", "HD21", "1H", "FE"]))
    ['C' 'C' 'C' 'O' 'H' 'H' 'FE']

    """
    if isinstance(atoms, (AtomArray, AtomArrayStack)):
        atom_names = atoms.atom_name
    else:
        atom_names = atoms
    return np.array([_guess_element(name) for name in atom_names])


def create_atom_names(atoms):
    """
    Create atom names for a single residue based on elements.

    The atom names are simply enumerated separately for each element.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack or array-like of str
        The atoms for which the atom names should be created.
        Alternatively the elements can be passed directly.

    Returns
    -------
    atom_names : ndarray, dtype=str
        The atom names.

    See Also
    --------
    infer_elements : The opposite of this function

    Notes
    -----
    The atom names created this way may differ from the ones in the
    original source, as different schemes for atom naming exist.
    This function only ensures that the created atom names are unique.
    This is e.g. necessary for writing bonds to PDBx files.

    Note that this function should be used only on single residues,
    otherwise enumeration would continue in the next residue.

    Examples
    --------

    >>> atoms = residue("URA")  # Uracil
    >>> print(atoms.element)
    ['N' 'C' 'O' 'N' 'C' 'O' 'C' 'C' 'H' 'H' 'H' 'H']
    >>> print(create_atom_names(atoms))
    ['N1' 'C1' 'O1' 'N2' 'C2' 'O2' 'C3' 'C4' 'H1' 'H2' 'H3' 'H4']
    """
    if isinstance(atoms, (AtomArray, AtomArrayStack)):
        elements = atoms.element
    else:
        elements = atoms

    atom_names = np.zeros(len(elements), dtype="U6")
    element_counter = Counter()
    for i, elem in enumerate(elements):
        element_counter[elem] += 1
        atom_names[i] = f"{elem}{element_counter[elem]}"
    return atom_names


_elements = [elem.upper() for elem in
["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
"Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
"Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
"Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
"I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
"Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
"Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
"U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
"Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts",
"Og"]
]
def _guess_element(atom_name):
    # remove digits (1H -> H)
    elem = "".join([i for i in atom_name if not i.isdigit()])
    elem = elem.upper()
    if len(elem) == 0:
        return ""

    # Some often used elements for biomolecules
    if elem.startswith("C") or elem.startswith("N") or \
        elem.startswith("O") or elem.startswith("S") or \
        elem.startswith("H"):
        return elem[0]

    # Exactly match element abbreviations
    try:
        return _elements[_elements.index(elem[:2])]
    except ValueError:
        try:
            return _elements[_elements.index(elem[0])]
        except ValueError:
            warnings.warn(f"Could not infer element for '{atom_name}'")
            return ""