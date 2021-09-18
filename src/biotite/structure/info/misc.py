# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["all_residues", "full_name", "link_type"]

from os.path import join, dirname, realpath
import msgpack


_info_dir = dirname(realpath(__file__))
# Data is taken from
# ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
# (2019/01/27)
with open(join(_info_dir, "residue_names.msgpack"), "rb") as file:
    _res_names = msgpack.load(file, raw=False)
with open(join(_info_dir, "link_types.msgpack"), "rb") as file:
    _link_types = msgpack.load(file, raw=False)


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
    ['0Y4', '0Y5', '0Y7', '0Y8', '0Y9', '0YA', '0YB', '0YC', '0YD', '0YE']
    """
    return list(_res_names.keys()) 


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
    return _res_names.get(res_name.upper())


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
    return _link_types.get(res_name.upper())