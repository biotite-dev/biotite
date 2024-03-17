# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["all_residues", "full_name", "link_type"]

from .ccd import get_ccd, get_from_ccd


def all_residues():
    """
    Get a list of all residues/compound names in the
    PDB chemical components dictionary.

    Returns
    -------
    residues : list of str
        A list of all available The up to 3-letter residue names.

    Examples
    --------

    >>> print(all_residues()[1000 : 1010])
    ['0V9', '0VA', '0VB', '0VC', '0VD', '0VE', '0VF', '0VG', '0VH', '0VI']
    """
    return get_ccd()["chem_comp"]["id"].as_array().tolist()


def full_name(res_name):
    """
    Get the full name of a residue/compound from the up to 3-letter
    residue name, based on the PDB chemical components dictionary.

    Parameters
    ----------
    res_name : str
        The up to 3-letter residue name.

    Returns
    -------
    name : str
        The full name of the residue.

    Examples
    --------

    >>> print(full_name("MAN"))
    alpha-D-mannopyranose
    """
    return get_from_ccd("chem_comp", res_name.upper(), "name").item()


def link_type(res_name):
    """
    Get the linking type of a residue/compound,
    based on the PDB chemical components dictionary.

    Parameters
    ----------
    res_name : str
        The up to 3-letter residue name.

    Returns
    -------
    link_type : str
        The link type.

    Examples
    --------

    >>> print(link_type("MAN"))
    D-saccharide, alpha linking
    >>> print(link_type("TRP"))
    L-PEPTIDE LINKING
    >>> print(link_type("HOH"))
    NON-POLYMER
    """
    return get_from_ccd("chem_comp", res_name.upper(), "type").item()