# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["amino_acid_names", "nucleotide_names", "carbohydrate_names"]

import functools
from pathlib import Path

CCD_DIR = Path(__file__).parent / "ccd"


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
    return _get_group_members("amino_acids")


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
    return _get_group_members("nucleotides")


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
    return _get_group_members("carbohydrates")


def _get_group_members(group_name):
    with open(CCD_DIR / f"{group_name}.txt", "r") as file:
        return tuple(file.read().split())
