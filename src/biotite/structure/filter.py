# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility functions for creating filters on atom
arrays and atom array stacks.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann, Tom David Müller"
__all__ = [
    "filter_solvent",
    "filter_monoatomic_ions",
    "filter_nucleotides",
    "filter_canonical_nucleotides",
    "filter_amino_acids",
    "filter_canonical_amino_acids",
    "filter_carbohydrates",
    "filter_intersection",
    "filter_first_altloc",
    "filter_highest_occupancy_altloc",
    "filter_peptide_backbone",
    "filter_phosphate_backbone",
    "filter_linear_bond_continuity",
    "filter_polymer",
]


from functools import partial
import numpy as np
from biotite.structure.atoms import array as atom_array
from biotite.structure.info.groups import (
    amino_acid_names,
    carbohydrate_names,
    nucleotide_names,
)
from biotite.structure.residues import get_residue_count, get_residue_starts

_canonical_aa_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "PYL",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
]
_canonical_nucleotide_list = ["A", "DA", "G", "DG", "C", "DC", "U", "DT"]

_solvent_list = ["HOH", "SOL"]

_peptide_backbone_atoms = ["N", "CA", "C"]
_phosphate_backbone_atoms = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]


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
    return array.res_name == array.element


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
    return np.isin(array.res_name, _solvent_list)


def filter_canonical_nucleotides(array):
    """
    Filter all atoms of one array that belong to canonical nucleotides.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        belongs to a canonical nucleotide.
    """
    return np.isin(array.res_name, _canonical_nucleotide_list)


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

    Notes
    -----
    Nucleotides are identified according to the PDB chemical component
    dictionary. A residue is considered a nucleotide if it its
    ``_chem_comp.type`` property has one of the following values (case
    insensitive):

    ``DNA LINKING``, ``DNA OH 3 PRIME TERMINUS``,
    ``DNA OH 5 PRIME TERMINUS``, ``L-DNA LINKING``, ``L-RNA LINKING``,
    ``RNA LINKING``, ``RNA OH 3 PRIME TERMINUS``,
    ``RNA OH 5 PRIME TERMINUS``
    """
    return np.isin(array.res_name, nucleotide_names())


def filter_canonical_amino_acids(array):
    """
    Filter all atoms of one array that belong to canonical amino acid
    residues.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        belongs to a canonical amino acid residue.
    """
    return np.isin(array.res_name, _canonical_aa_list)


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

    Notes
    -----
    Amino acids are identified according to the PDB chemical component
    dictionary. A residue is considered an amino acid if it its
    ``_chem_comp.type`` property has one of the following values (case
    insensitive):

    ``D-BETA-PEPTIDE``, ``C-GAMMA LINKING``, ``D-GAMMA-PEPTIDE``,
    ``C-DELTA LINKING``, ``D-PEPTIDE LINKING``,
    ``D-PEPTIDE NH3 AMINO TERMINUS``,
    ``L-BETA-PEPTIDE, C-GAMMA LINKING``,
    ``L-GAMMA-PEPTIDE, C-DELTA LINKING``,
    ``L-PEPTIDE COOH CARBOXY TERMINUS``, ``L-PEPTIDE LINKING``,
    ``L-PEPTIDE NH3 AMINO TERMINUS``, ``PEPTIDE LINKING``
    """
    return np.isin(array.res_name, amino_acid_names())


def filter_carbohydrates(array):
    """
    Filter all atoms of one array that belong to carbohydrates.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where the atom
        belongs to a carbohydrate.

    Notes
    -----
    Carbohydrates are identified according to the PDB chemical component
    dictionary. A residue is considered a carbohydrate if it its
    ``_chem_comp.type`` property has one of the following values (case
    insensitive):

    ``D-SACCHARIDE``, ``D-SACCHARIDE,ALPHA LINKING``,
    ``D-SACCHARIDE, BETA LINKING``, ``L-SACCHARIDE``,
    ``L-SACCHARIDE, ALPHA LINKING``, ``L-SACCHARIDE, BETA LINKING``,
    ``SACCHARIDE``
    """
    return np.isin(array.res_name, carbohydrate_names())


def _filter_atom_names(array, atom_names):
    return np.isin(array.atom_name, atom_names)


def filter_peptide_backbone(array):
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
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return _filter_atom_names(array, _peptide_backbone_atoms) & filter_amino_acids(
        array
    )


def filter_phosphate_backbone(array):
    """
    Filter all phosphate backbone atoms of one array.

    This includes the P, O5', C5', C4', C3', and O3' atoms.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is ``True`` for all indices in `array`, where an atom
        is a part of the phosphate backbone.
    """

    return _filter_atom_names(array, _phosphate_backbone_atoms) & filter_nucleotides(
        array
    )


def filter_linear_bond_continuity(array, min_len=1.2, max_len=1.8):
    """
    Filter for atoms such that their bond length with the next atom
    lies within the provided boundaries.

    The result will depend on the atoms' order.
    For instance, consider a molecule:

    .. code-block:: none

           C3
           |
        C1-C2-C4

    If the order corresponds to ``[C1, C2, C4, C3]``, the output will be
    ``[True, True, False, True]``.
    Note that the trailing atom will always evaluate to ``True``.

    Parameters
    ----------
    array : AtomArray
        The array to filter.
    min_len : float
        Minmum bond length.
    max_len : float
        Maximum bond length.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        has a bond length with the next atom within [`min_len`, `max_len`]
        boundaries.

    Notes
    -----
    Note that this function purely uses distances between consecutive atoms.
    A potential ``BondList`` is not considered here.
    """
    dist = np.linalg.norm(np.diff(array.coord, axis=0), axis=1)
    mask = (dist >= min_len) & (dist <= max_len)
    return np.append(mask, True)


def _is_polymer(array, min_size, pol_type):
    if pol_type.startswith("p"):
        filt_fn = filter_amino_acids
    elif pol_type.startswith("n"):
        filt_fn = filter_nucleotides
    elif pol_type.startswith("c"):
        filt_fn = filter_carbohydrates
    else:
        raise ValueError(f"Unsupported polymer type {pol_type}")

    mask = filt_fn(array)
    return get_residue_count(array[mask]) >= min_size


def filter_polymer(array, min_size=2, pol_type="peptide"):
    """
    Filter for atoms that are a part of a consecutive standard macromolecular
    polymer entity.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to filter.
    min_size : int
        The minimum number of monomers.
    pol_type : str
        The polymer type, either ``"peptide"``, ``"nucleotide"``, or ``"carbohydrate"``.
        Abbreviations are supported: ``"p"``, ``"pep"``, ``"n"``, etc.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where atoms belong to
        consecutive polymer entity having at least `min_size` monomers.
    """
    # Import `check_res_id_continuity` here to avoid circular imports
    from biotite.structure.integrity import check_res_id_continuity

    split_idx = check_res_id_continuity(array)

    check_pol = partial(_is_polymer, min_size=min_size, pol_type=pol_type)
    bool_idx = map(
        lambda a: np.full(len(a), check_pol(atom_array(a)), dtype=bool),
        np.split(array, split_idx),
    )
    return np.concatenate(list(bool_idx))


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
    categories = [
        category
        for category in array.get_annotation_categories()
        if category in intersect_categories
    ]
    for i in range(array.array_length()):
        subfilter = np.full(intersect.array_length(), True, dtype=bool)
        for category in categories:
            subfilter &= (
                intersect.get_annotation(category) == array.get_annotation(category)[i]
            )
        filter[i] = subfilter.any()
    return filter


def filter_first_altloc(atoms, altloc_ids):
    """
    Filter all atoms, that have the first *altloc* ID appearing in a
    residue.

    Structure files (PDB, PDBx) allow for duplicate atom records,
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
    altloc_filter = np.isin(altloc_ids, [".", "?", " ", ""])

    # And filter all atoms for each residue with the first altloc ID
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        letter_altloc_ids = [loc for loc in altloc_ids[start:stop] if loc.isalpha()]
        if len(letter_altloc_ids) > 0:
            first_id = letter_altloc_ids[0]
            altloc_filter[start:stop] |= altloc_ids[start:stop] == first_id
        else:
            # No altloc ID in this residue -> Nothing to do
            pass

    return altloc_filter


def filter_highest_occupancy_altloc(atoms, altloc_ids, occupancies):
    """
    For each residue, filter all atoms, that have the *altloc* ID
    with the highest occupancy for this residue.

    Structure files (PDB, PDBx) allow for duplicate atom records,
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
    altloc_filter = np.isin(altloc_ids, [".", "?", " ", ""])

    # And filter all atoms for each residue with the highest sum of
    # occupancies
    residue_starts = get_residue_starts(atoms, add_exclusive_stop=True)
    for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
        occupancies_in_res = occupancies[start:stop]
        altloc_ids_in_res = altloc_ids[start:stop]

        letter_altloc_ids = [loc for loc in altloc_ids_in_res if loc.isalpha()]

        if len(letter_altloc_ids) > 0:
            highest = -1.0
            highest_id = None
            for id in sorted(set(letter_altloc_ids)):
                occupancy_sum = np.sum(occupancies_in_res[altloc_ids_in_res == id])
                if occupancy_sum > highest:
                    highest = occupancy_sum
                    highest_id = id
            altloc_filter[start:stop] |= altloc_ids[start:stop] == highest_id
        else:
            # No altloc ID in this residue -> Nothing to do
            pass

    return altloc_filter
