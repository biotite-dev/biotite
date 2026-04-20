# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["amino_acid_names", "nucleotide_names", "carbohydrate_names", "ion_names"]

import functools
import numpy as np
from biotite.structure.info.ccd import get_ccd, get_from_ccd

_AMINO_ACID_TYPES = [
    "D-beta-peptide, C-gamma linking",
    "D-gamma-peptide, C-delta linking",
    "D-peptide COOH carboxy terminus",
    "D-peptide NH3 amino terminus",
    "D-peptide linking",
    "L-beta-peptide, C-gamma linking",
    "L-gamma-peptide, C-delta linking",
    "L-peptide COOH carboxy terminus",
    "L-peptide NH3 amino terminus",
    "L-peptide linking",
    "peptide linking",
]
_NUCLEOTIDE_TYPES = [
    "DNA OH 3 prime terminus",
    "DNA OH 5 prime terminus",
    "DNA linking",
    "L-DNA linking",
    "L-RNA linking",
    "RNA OH 3 prime terminus",
    "RNA OH 5 prime terminus",
    "RNA linking",
]
_CARBOHYDRATE_TYPES = [
    "D-saccharide",
    "D-saccharide, alpha linking",
    "D-saccharide, beta linking",
    "L-saccharide",
    "L-saccharide, alpha linking",
    "L-saccharide, beta linking",
    "saccharide",
]


@functools.cache
def amino_acid_names():
    """
    Get a tuple of all amino acid residues according to the PDB
    *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    amino_acid_names : tuple of str
        A list of codes containing residues that are
        peptide monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_AMINO_ACID_TYPES)


@functools.cache
def nucleotide_names():
    """
    Get a tuple of all nucleotide residues according to the
    PDB *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    nucleotide_names : tuple of str
        A list of codes containing residues that are
        DNA/RNA monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_NUCLEOTIDE_TYPES)


@functools.cache
def carbohydrate_names():
    """
    Get a tuple of all carbohydrate residues according to the
    PDB *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    carbohydrate_names : tuple of str
        A list of codes containing residues that are
        saccharide monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_CARBOHYDRATE_TYPES)


@functools.cache
def ion_names():
    """
    Get a tuple of all residues in the
    PDB *Chemical Component Dictionary* (CCD) that are monoatomic ions.
    :footcite:`Westbrook2015`

    Returns
    -------
    ion_names : tuple of str
        A list of codes containing residues that are monoatomic ions.

    References
    ----------

    .. footbibliography::
    """
    chem_comp = get_ccd()["chem_comp"]
    comp_ids = chem_comp["id"].as_array()
    formulas = chem_comp["formula"].as_array()

    # Simple filter: The formula must contain only one element
    mask = np.char.isalpha(formulas)
    comp_ids = comp_ids[mask]
    formulas = formulas[mask]

    # More complex filter: The component must only contain a single charged atom
    ion_comp_ids = []
    for comp_id in comp_ids.tolist():
        chem_comp_atom = get_from_ccd("chem_comp_atom", comp_id)
        if chem_comp_atom.row_count != 1:
            continue
        if chem_comp_atom["charge"].as_item() == 0:
            continue
        ion_comp_ids.append(comp_id)

    return ion_comp_ids


def _get_group_members(match_types):
    """
    Identify component IDs that matches a given component *type* from the CCD.

    Parameters
    ----------
    match_types : list of str
        The component types to extract.

    Returns
    -------
    comp_ids : list of str
        The extracted component IDs.
    """
    category = get_ccd()["chem_comp"]
    comp_ids = category["id"].as_array()
    types = category["type"].as_array()
    # Ignore case
    return comp_ids[np.isin(np.char.lower(types), np.char.lower(match_types))].tolist()
