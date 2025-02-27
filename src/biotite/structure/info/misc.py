# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["all_residues", "full_name", "link_type", "one_letter_code"]

from biotite.structure.info.ccd import get_ccd, get_from_ccd


def all_residues():
    """
    Get a list of all residues/compound names in the PDB
    *Chemical Component Dictionary* (CCD).

    Returns
    -------
    residues : list of str
        A list of all available residue names.
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
    name : str or None
        The full name of the residue.
        If the residue is unknown to the chemical components dictionary,
        ``None`` is returned.

    Examples
    --------

    >>> print(full_name("MAN"))
    alpha-D-mannopyranose
    """
    column = get_from_ccd("chem_comp", res_name.upper(), "name")
    if column is None:
        return None
    return column.as_item()


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
    link_type : str or None
        The link type.
        If the residue is unknown to the chemical components dictionary,
        ``None`` is returned.

    Examples
    --------

    >>> print(link_type("MAN"))
    D-saccharide, alpha linking
    >>> print(link_type("TRP"))
    L-PEPTIDE LINKING
    >>> print(link_type("HOH"))
    NON-POLYMER
    """
    column = get_from_ccd("chem_comp", res_name.upper(), "type")
    if column is None:
        return None
    return column.as_item()


def one_letter_code(res_name):
    """
    Get the one-letter code of a residue/compound,
    based on the PDB chemical components dictionary.

    The one-letter code is only defined for amino acids and nucleotides
    and for compounds that are structurally similar to them.

    Parameters
    ----------
    res_name : str
        The up to 3-letter residue name.

    Returns
    -------
    one_letter_code : str or None
        The one-letter code.
        ``None`` if the compound is not present in the CCD or if no
        one-letter code is defined for this compound.

    Examples
    --------

    Get the one letter code for an amino acid (or a nucleotide).

    >>> print(full_name("ALA"))
    ALANINE
    >>> print(one_letter_code("ALA"))
    A

    For similar compounds, the one-letter code is also defined.

    >>> print(full_name("DAL"))
    D-ALANINE
    >>> print(one_letter_code("DAL"))
    A

    For other compounds, the one-letter code is not defined.

    >>> print(full_name("MAN"))
    alpha-D-mannopyranose
    >>> print(one_letter_code("MAN"))
    None
    """
    column = get_from_ccd("chem_comp", res_name.upper(), "one_letter_code")
    if column is None:
        return None
    if column.mask is not None:
        # Value is masked, i.e. inapplicable or missing
        return None
    return column.as_item()
