# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["amino_acid_names", "nucleotide_names", "carbohydrate_names"]

from pathlib import Path
import copy


CCD_DIR = Path(__file__).parent / "ccd"


group_lists = {}


def amino_acid_names():
    """
    Get a tuple of amino acid three-letter codes according to the
    PDB *Chemical Component Dictionary* :footcite:`Westbrook2015`.

    Returns
    -------
    amino_acid_names : tuple of str
        A list of three-letter-codes containing residues that are
        peptide monomers.

    Notes
    -----

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members("amino_acids")


def nucleotide_names():
    """
    Get a tuple of nucleotide three-letter codes according to the
    PDB *Chemical Component Dictionary* :footcite:`Westbrook2015`.

    Returns
    -------
    nucleotide_names : tuple of str
        A list of three-letter-codes containing residues that are
        DNA/RNA monomers.

    Notes
    -----

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members("nucleotides")


def carbohydrate_names():
    """
    Get a tuple of carbohydrate three-letter codes according to the
    PDB *Chemical Component Dictionary* :footcite:`Westbrook2015`.

    Returns
    -------
    carbohydrate_names : tuple of str
        A list of three-letter-codes containing residues that are
        saccharide monomers.

    Notes
    -----

    References
    ----------

    .. footbibliography::
    """
    return _get_group_members("carbohydrates")


def _get_group_members(group_name):
    global group_lists
    if group_name not in group_lists:
        with open(CCD_DIR / f"{group_name}.txt", "r") as file:
            group_lists[group_name] = tuple(file.read().split())
    return group_lists[group_name]