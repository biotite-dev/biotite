# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["amino_acid_names", "nucleotide_names", "carbohydrate_names"]

import functools
import numpy as np
from biotite.structure.info.ccd import get_ccd

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
    Get a tuple of amino acid three-letter codes according to the
    PDB *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    amino_acid_names : tuple of str
        A list of three-letter-codes containing residues that are
        peptide monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_AMINO_ACID_TYPES)


@functools.cache
def nucleotide_names():
    """
    Get a tuple of nucleotide three-letter codes according to the
    PDB *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    nucleotide_names : tuple of str
        A list of three-letter-codes containing residues that are
        DNA/RNA monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_NUCLEOTIDE_TYPES)


@functools.cache
def carbohydrate_names():
    """
    Get a tuple of carbohydrate three-letter codes according to the
    PDB *Chemical Component Dictionary*.
    :footcite:`Westbrook2015`

    Returns
    -------
    carbohydrate_names : tuple of str
        A list of three-letter-codes containing residues that are
        saccharide monomers.

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members(_CARBOHYDRATE_TYPES)


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
