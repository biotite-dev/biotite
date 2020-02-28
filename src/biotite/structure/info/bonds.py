# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["bond_dataset", "bond_order", "bonds_in_residue"]

import msgpack
import copy
from os.path import join, dirname, realpath


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
        _intra_bonds_raw = msgpack.unpack(
            file, use_list=False, raw=False, strict_map_key=False
        )
        _intra_bonds = {}
        for group, group_bonds_raw in _intra_bonds_raw.items():
            group_bonds = {
                frozenset(bond_raw) : count
                for bond_raw, count in group_bonds_raw.items()
            }
            _intra_bonds[group] = group_bonds


def bond_dataset():
    """
    Get a copy of the complete bond dataset extracted from the chemical 
    compound dictionary.

    This dataset does only contain intra-residue bonds.

    Returns
    -------
    bonds : dict
        The bonds as nested dictionary.
        It maps residue names (up to 3-letters, upper case) to
        inner dictionaries.
        Each of these dictionary contains the bond information for the
        given residue.
        Specifically, it uses a set of two atom names, that are bonded,
        as keys and the respective bond order as values.
    """
    _init_dataset()
    return copy.copy(_intra_bonds)


def bond_order(res_name, atom_name1, atom_name2):
    """
    Get the bond order for two atoms of the same residue, based
    on the PDB chemical compound dictionary.

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
        to the chemical compound dictionary, `None` is returned.
    
    Examples
    --------

    >>> print(bond_order("ALA", "CA", "CB"))
    1
    >>> print(bond_order("ALA", "C", "O"))
    2
    >>> print(bond_order("ALA", "C", "CB"))
    None
    >>> print(bond_order("ALA", "FOO", "BAR"))
    None
    """
    _init_dataset()
    group_bonds = _intra_bonds.get(res_name.upper())
    if group_bonds is None:
        return None
    else:
        return group_bonds.get(frozenset((atom_name1, atom_name2)))


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
    bonds : dict
        A dictionary that maps sets of two atom names to their
        respective bond orders. `None` if the residue is unknown to the
        chemical compound dictionary.
    
    Examples
    --------
    >>> bonds = bonds_in_residue("MAN")
    >>> for atoms, order in sorted(bonds.items()):
    ...     atom1, atom2 = sorted(atoms)
    ...     print(f"{atom1:3} + {atom2:3} -> {order}")
    C1  + C2  -> 1
    C1  + O1  -> 1
    C1  + O5  -> 1
    C1  + H1  -> 1
    C2  + C3  -> 1
    C2  + O2  -> 1
    C2  + H2  -> 1
    C3  + C4  -> 1
    C3  + O3  -> 1
    C3  + H3  -> 1
    C4  + C5  -> 1
    C4  + O4  -> 1
    C4  + H4  -> 1
    C5  + C6  -> 1
    C5  + O5  -> 1
    C5  + H5  -> 1
    C6  + O6  -> 1
    C6  + H61 -> 1
    C6  + H62 -> 1
    HO1 + O1  -> 1
    HO2 + O2  -> 1
    HO3 + O3  -> 1
    HO4 + O4  -> 1
    HO6 + O6  -> 1
    """
    _init_dataset()
    return copy.copy(_intra_bonds.get(res_name.upper()))