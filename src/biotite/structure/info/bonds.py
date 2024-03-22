# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["bond_type", "bonds_in_residue"]

from ..bonds import BondType
from .ccd import get_from_ccd


BOND_TYPES = {
    ("SING", "N") : BondType.SINGLE,
    ("DOUB", "N") : BondType.DOUBLE,
    ("TRIP", "N") : BondType.TRIPLE,
    ("QUAD", "N") : BondType.QUADRUPLE,
    ("SING", "Y") : BondType.AROMATIC_SINGLE,
    ("DOUB", "Y") : BondType.AROMATIC_DOUBLE,
    ("TRIP", "Y") : BondType.AROMATIC_TRIPLE,
}

_intra_bonds = {}


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
    bonds_for_residue = bonds_in_residue(res_name)
    if bonds_for_residue is None:
        return None
    # Try both atom orders
    bond_type_int = bonds_for_residue.get(
        (atom_name1, atom_name2),
        bonds_for_residue.get((atom_name2, atom_name1))
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
        Empty, if the residue is unknown to the
        chemical components dictionary.

    Warnings
    --------
    Treat the returned dictionary as immutable.
    Modifying the dictionary may lead to unexpected behavior.
    In other functionalities throughout *Biotite* that uses this
    function.

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
    global _intra_bonds
    if res_name not in _intra_bonds:
        chem_comp_bond_dict = get_from_ccd("chem_comp_bond", res_name)
        if chem_comp_bond_dict is None:
            _intra_bonds[res_name] = {}
        else:
            bonds_for_residue = {}
            for atom1, atom2, order, aromatic_flag in zip(
                chem_comp_bond_dict["atom_id_1"],
                chem_comp_bond_dict["atom_id_2"],
                chem_comp_bond_dict["value_order"],
                chem_comp_bond_dict["pdbx_aromatic_flag"]
            ):
                bond_type = BOND_TYPES[order, aromatic_flag]
                bonds_for_residue[atom1.item(), atom2.item()] = bond_type
            _intra_bonds[res_name] = bonds_for_residue
    return _intra_bonds[res_name]
