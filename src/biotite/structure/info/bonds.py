# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["bond_dataset", "bond_order", "bond_type", "bonds_in_residue"]

import warnings
import copy
from os.path import join, dirname, realpath
import msgpack
from ..bonds import BondType


_intra_bonds = None


def _init_dataset():
    """
    Load the bond dataset from MessagePack file.

    Since loading the database is computationally expensive,
    this is only done, when the bond database is actually required.
    """
    global _intra_bonds
    if _intra_bonds is not None:
        # Database is already initialized
        return

    # Bonds are taken from
    # ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
    # (2019/01/27)
    _info_dir = dirname(realpath(__file__))
    with open(join(_info_dir, "intra_bonds.msgpack"), "rb") as file:
        _intra_bonds= msgpack.unpack(
            file, use_list=False, raw=False, strict_map_key=False
        )


def bond_dataset():
    """
    Get a copy of the complete bond dataset extracted from the chemical 
    components dictionary.

    This dataset does only contain intra-residue bonds.

    Returns
    -------
    bonds : dict (str -> dict ((str, str) -> int))
        The bonds as nested dictionary.
        It maps residue names (up to 3-letters, upper case) to
        inner dictionaries.
        Each of these dictionary contains the bond information for the
        given residue.
        Specifically, it uses a set of two atom names, that are bonded,
        as keys and the respective :class:`BondType`
        (represented by an integer) as values.
    """
    _init_dataset()
    return copy.copy(_intra_bonds)


def bond_order(res_name, atom_name1, atom_name2):
    """
    Get the bond order for two atoms of the same residue, based
    on the PDB chemical components dictionary.

    DEPRECATED: Please use :func:`bond_type()` instead.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue
        `atom_name1` and `atom_name2` belong to.
    atom_name1, atom_name2 : str
        The names of the two atoms to get the bond order from.
    
    Returns
    -------
    order : int or None
        The order of the bond between `atom_name1` and `atom_name2`.
        If the atoms form no bond, if any of the two atoms does not
        exist in the context of the residue or if the residue is unknown
        to the chemical components dictionary, `None` is returned.
    """
    warnings.warn("Please use `bond_type()` instead", DeprecationWarning)

    _init_dataset()
    btype = bond_type(res_name, atom_name1, atom_name2)
    if btype is None:
        return None
    elif btype == BondType.AROMATIC_SINGLE:
        return 1
    elif btype == BondType.AROMATIC_DOUBLE:
        return 2
    else:
        return int(btype)


def bond_type(res_name, atom_name1, atom_name2):
    """
    Get the :class:`BondType` for two atoms of the same residue, based
    on the PDB chemical components dictionary.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue
        `atom_name1` and `atom_name2` belong to.
    atom_name1, atom_name2 : str
        The names of the two atoms to get the bond order from.
    
    Returns
    -------
    order : BondType or None
        The :class:`BondType` of the bond between `atom_name1` and
        `atom_name2`.
        If the atoms form no bond, if any of the two atoms does not
        exist in the context of the residue or if the residue is unknown
        to the chemical components dictionary, `None` is returned.
    
    Examples
    --------

    >>> print(bond_type("PHE", "CA", "CB"))
    BondType.SINGLE
    >>> print(bond_type("PHE", "CG", "CD1"))
    BondType.AROMATIC_DOUBLE
    >>> print(bond_type("PHE", "CA", "CG"))
    None
    >>> print(bond_type("PHE", "FOO", "BAR"))
    None
    """
    _init_dataset()
    group_bonds = _intra_bonds.get(res_name.upper())
    if group_bonds is None:
        return None
    # Try both atom aroders
    bond_type_int = group_bonds.get(
        (atom_name1, atom_name2),
        group_bonds.get((atom_name2, atom_name1))
    )
    if bond_type_int is not None:
        return BondType(bond_type_int)
    else:
        return None


def bonds_in_residue(res_name):
    """
    Get a dictionary containing all atoms inside a given residue
    that form a bond.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue to get the bonds for.
    
    Returns
    -------
    bonds : dict (str -> int)
        A dictionary that maps tuples of two atom names to their
        respective bond types (represented as integer).
        `None` if the residue is unknown to the
        chemical components dictionary.
    
    Examples
    --------
    >>> bonds = bonds_in_residue("PHE")
    >>> for atoms, bond_type_int in sorted(bonds.items()):
    ...     atom1, atom2 = sorted(atoms)
    ...     print(f"{atom1:3} + {atom2:3} -> {str(BondType(bond_type_int))}")
    C   + O   -> BondType.DOUBLE
    C   + OXT -> BondType.SINGLE
    C   + CA  -> BondType.SINGLE
    CA  + CB  -> BondType.SINGLE
    CA  + HA  -> BondType.SINGLE
    CB  + CG  -> BondType.SINGLE
    CB  + HB2 -> BondType.SINGLE
    CB  + HB3 -> BondType.SINGLE
    CD1 + CE1 -> BondType.AROMATIC_SINGLE
    CD1 + HD1 -> BondType.SINGLE
    CD2 + CE2 -> BondType.AROMATIC_DOUBLE
    CD2 + HD2 -> BondType.SINGLE
    CE1 + CZ  -> BondType.AROMATIC_DOUBLE
    CE1 + HE1 -> BondType.SINGLE
    CE2 + CZ  -> BondType.AROMATIC_SINGLE
    CE2 + HE2 -> BondType.SINGLE
    CD1 + CG  -> BondType.AROMATIC_DOUBLE
    CD2 + CG  -> BondType.AROMATIC_SINGLE
    CZ  + HZ  -> BondType.SINGLE
    CA  + N   -> BondType.SINGLE
    H   + N   -> BondType.SINGLE
    H2  + N   -> BondType.SINGLE
    HXT + OXT -> BondType.SINGLE
    """
    _init_dataset()
    return copy.copy(_intra_bonds.get(res_name.upper()))