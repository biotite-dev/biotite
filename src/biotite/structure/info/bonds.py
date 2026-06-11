# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["bond_type", "bonds_in_residue"]

import functools
from biotite.structure.bonds import BondType
from biotite.structure.info.ccd import get_from_ccd

BOND_TYPES = {
    ("SING", "N"): BondType.SINGLE,
    ("DOUB", "N"): BondType.DOUBLE,
    ("TRIP", "N"): BondType.TRIPLE,
    ("QUAD", "N"): BondType.QUADRUPLE,
    ("SING", "Y"): BondType.AROMATIC_SINGLE,
    ("DOUB", "Y"): BondType.AROMATIC_DOUBLE,
    ("TRIP", "Y"): BondType.AROMATIC_TRIPLE,
}


def bond_type(res_name: str, atom_name1: str, atom_name2: str) -> BondType | None:
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

    >>> print(repr(bond_type("PHE", "CA", "CB")))
    <BondType.SINGLE: 1>
    >>> print(repr(bond_type("PHE", "CG", "CD1")))
    <BondType.AROMATIC_DOUBLE: 6>
    >>> print(repr(bond_type("PHE", "CA", "CG")))
    None
    >>> print(repr(bond_type("PHE", "FOO", "BAR")))
    None
    """
    bonds_for_residue = bonds_in_residue(res_name)
    # Try both atom orders
    return bonds_for_residue.get(
        (atom_name1, atom_name2), bonds_for_residue.get((atom_name2, atom_name1))
    )


@functools.lru_cache(maxsize=1000)
def bonds_in_residue(res_name: str) -> dict[tuple[str, str], BondType]:
    """
    Get a dictionary containing all atoms inside a given residue
    that form a bond.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue to get the bonds for.

    Returns
    -------
    bonds : dict ((str, str) -> BondType)
        A dictionary that maps tuples of two atom names to their
        respective bond types.
        Empty, if the residue is unknown to the
        chemical components dictionary.

    Warnings
    --------
    Treat the returned dictionary as immutable.
    Modifying the dictionary may lead to unexpected behavior.
    In other functionalities throughout *Biotite* that uses this
    function.

    Notes
    -----
    The returned values are cached for faster access in subsequent calls.

    Examples
    --------
    >>> bonds = bonds_in_residue("PHE")
    >>> for atoms, bond_type_int in sorted(bonds.items()):
    ...     atom1, atom2 = sorted(atoms)
    ...     print(f"{atom1:3} + {atom2:3} -> {BondType(bond_type_int).name}")
    C   + O   -> DOUBLE
    C   + OXT -> SINGLE
    C   + CA  -> SINGLE
    CA  + CB  -> SINGLE
    CA  + HA  -> SINGLE
    CB  + CG  -> SINGLE
    CB  + HB2 -> SINGLE
    CB  + HB3 -> SINGLE
    CD1 + CE1 -> AROMATIC_SINGLE
    CD1 + HD1 -> SINGLE
    CD2 + CE2 -> AROMATIC_DOUBLE
    CD2 + HD2 -> SINGLE
    CE1 + CZ  -> AROMATIC_DOUBLE
    CE1 + HE1 -> SINGLE
    CE2 + CZ  -> AROMATIC_SINGLE
    CE2 + HE2 -> SINGLE
    CD1 + CG  -> AROMATIC_DOUBLE
    CD2 + CG  -> AROMATIC_SINGLE
    CZ  + HZ  -> SINGLE
    CA  + N   -> SINGLE
    H   + N   -> SINGLE
    H2  + N   -> SINGLE
    HXT + OXT -> SINGLE
    """
    chem_comp_bond = get_from_ccd("chem_comp_bond", res_name)
    if chem_comp_bond is None:
        return {}
    else:
        bonds_for_residue = {}
        for atom1, atom2, order, aromatic_flag in zip(
            chem_comp_bond["atom_id_1"].as_array(),
            chem_comp_bond["atom_id_2"].as_array(),
            chem_comp_bond["value_order"].as_array(),
            chem_comp_bond["pdbx_aromatic_flag"].as_array(),
        ):
            bond_type = BOND_TYPES[order, aromatic_flag]
            bonds_for_residue[atom1.item(), atom2.item()] = bond_type
        return bonds_for_residue
