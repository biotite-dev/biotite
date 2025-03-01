# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["vdw_radius_protor", "vdw_radius_single"]

from biotite.structure.info.bonds import bonds_in_residue

# fmt: off
# Contains tuples for the different ProtOr groups:
# Tuple contains: element, valency, H count
_PROTOR_RADII = {
    ("C",  3, 0) : 1.61,
    ("C",  3, 1) : 1.76,
    ("C",  4, 1) : 1.88,
    ("C",  4, 2) : 1.88,
    ("C",  4, 3) : 1.88,
    ("N",  3, 0) : 1.64,
    ("N",  3, 1) : 1.64,
    ("N",  3, 2) : 1.64,
    ("N",  4, 3) : 1.64,
    ("O",  1, 0) : 1.42,
    ("O",  2, 1) : 1.46,
    ("S",  1, 0) : 1.77,
    ("S",  2, 0) : 1.77, # Not official, added for completeness (MET)
    ("S",  2, 1) : 1.77,
    ("F",  1, 0) : 1.47, # Taken from _SINGLE_ATOM_VDW_RADII
    ("CL", 1, 0) : 1.75, # Taken from _SINGLE_ATOM_VDW_RADII
    ("BR", 1, 0) : 1.85, # Taken from _SINGLE_ATOM_VDW_RADII
    ("I",  1, 0) : 1.98, # Taken from _SINGLE_RADII
}

_SINGLE_ATOM_VDW_RADII = {
    # Main group
    # Row 1 (Period 1)
    "H":  1.10,
    "HE": 1.40,

    # Row 2 (Period 2)
    "LI": 1.81,
    "BE": 1.53,
    "B":  1.92,
    "C":  1.70,
    "N":  1.55,
    "O":  1.52,
    "F":  1.47,
    "NE": 1.54,

    # Row 3 (Period 3)
    "NA": 2.27,
    "MG": 1.73,
    "AL": 1.84,
    "SI": 2.10,
    "P":  1.80,
    "S":  1.80,
    "CL": 1.75,
    "AR": 1.88,

    # Row 4 (Period 4)
    "K":  2.75,
    "CA": 2.31,
    "GA": 1.87,
    "GE": 2.11,
    "AS": 1.85,
    "SE": 1.90,
    "BR": 1.83,
    "KR": 2.02,

    # Row 5 (Period 5)
    "RB": 3.03,
    "SR": 2.49,
    "IN": 1.93,
    "SN": 2.17,
    "SB": 2.06,
    "TE": 2.06,
    "I":  1.98,
    "XE": 2.16,

    # Row 6 (Period 6)
    "CS": 3.43,
    "BA": 2.68,
    "TL": 1.96,
    "PB": 2.02,
    "BI": 2.07,
    "PO": 1.97,
    "AT": 2.02,
    "RN": 2.20,

    # Row 7 (Period 7)
    "FR": 3.48,
    "RA": 2.83,

    # Transition metals (relevant ones only)
    # Row 1
    "FE": 2.05,
    "CU": 2.00,
    "ZN": 2.10,
    "MN": 2.05,
    "CO": 2.00,
    "NI": 2.00,

    # Row 2
    'MO': 2.10,
    'RU': 2.05,

    # Row 3
    'W': 2.10,
    'PT': 2.05,
    'AU': 2.10,
}
"""
Van der Waals radii for main group and transition elements.

Main group:
Source: https://pubs.acs.org/doi/10.1021/jp8111556, Table 12 (Mantina et al. 2009)

Transition metals:
Source: RDKit, 2024.9.4 Release
https://github.com/rdkit/rdkit/blob/af6347963f25cfe8fe4db0638410b2f3a8e8bd89/Code/GraphMol/atomic_data.cpp#L51

Where available, these values were cross-checked vs the CRC Handbook of
Chemistry and Physics (105th edition) and verified that they are closely
in line (barring very minor discrepancies, usually < 0.05 Å).
We cannot use the CRC values directly as they are not permissively licensed.
"""

# fmt: on

# A dictionary that caches radii for each residue
_protor_radii = {}


def vdw_radius_protor(res_name, atom_name):
    """
    Estimate the Van-der-Waals radius of a heavy atom,
    that includes the radius added by potential bonded hydrogen atoms.
    The respective radii are taken from the ProtOr dataset.
    :footcite:`Tsai1999`

    This is especially useful for macromolecular structures where no
    hydrogen atoms are resolved, e.g. crystal structures.
    The valency of the heavy atom and the amount of normally
    bonded hydrogen atoms is taken from the *Chemical Component Dictionary*.

    Parameters
    ----------
    res_name : str
        The up to 3-letter residue name the non-hydrogen atom belongs
        to.
    atom_name : str
        The name of the non-hydrogen atom.

    Returns
    -------
    radius : float
        The Van-der-Waals radius of the given atom.
        If the radius cannot be estimated for the atom, `None` is returned.

    See Also
    --------
    vdw_radius_single : *Van-der-Waals* radii for structures with annotated hydrogen atoms.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> print(vdw_radius_protor("GLY", "CA"))
    1.88
    """
    res_name = res_name.upper()
    if atom_name[0] == "H":
        raise ValueError(
            f"Calculating the ProtOr radius for the hydrogen atom "
            f"'{atom_name}' is not meaningful"
        )
    if res_name in _protor_radii:
        # Use cached radii for the residue, if already calculated
        if atom_name not in _protor_radii[res_name]:
            raise KeyError(
                f"Residue '{res_name}' does not contain an atom named '{atom_name}'"
            )
        return _protor_radii[res_name].get(atom_name)
    else:
        # Otherwise calculate radii for the given residue and cache
        _protor_radii[res_name] = _calculate_protor_radii(res_name)
        # Recursive call, but this time the radii for the given residue
        # are cached
        return vdw_radius_protor(res_name, atom_name)


def _calculate_protor_radii(res_name):
    """
    Calculate the ProtOr VdW radii for all atoms (atom names) in
    a residue.
    """
    bonds = bonds_in_residue(res_name)
    # Maps atom names to a ProtOr group
    # -> tuple(element, valency, H count)
    # Based on the group the radius is chosen from _PROTOR_RADII
    groups = {}
    for atom1, atom2 in bonds:
        # Process each bond two times:
        # One time the first atom is the one to get valency and H count
        # for and the other time vice versa
        for main_atom, bound_atom in ((atom1, atom2), (atom2, atom1)):
            element = main_atom[0]
            # Calculating ProtOr radii for hydrogens in not meaningful
            if element == "H":
                continue
            # Only for these elements ProtOr groups exist
            # Calculation of group for all other elements would be
            # pointless
            if element not in ["C", "N", "O", "S"]:
                # Empty tuple to indicate nonexistent entry
                groups[main_atom] = ()
                continue
            # Update existing entry if already existing
            group = groups.get(main_atom, [element, 0, 0])
            # Increase valency by one, since the bond entry exists
            group[1] += 1
            # If the atom is bonded to hydrogen, increase H count
            if bound_atom[0] == "H":
                group[2] += 1
            groups[main_atom] = group
    # Get radii based on ProtOr groups
    radii = {atom: _PROTOR_RADII.get(tuple(group)) for atom, group in groups.items()}
    return radii


def vdw_radius_single(element):
    """
    Get the *Van-der-Waals* radius of an atom from the given element.
    :footcite:`Mantina2009`

    Parameters
    ----------
    element : str
        The chemical element of the atoms.

    Returns
    -------
    radius : float
        The Van-der-Waals radius of the atom.
        If the radius is unknown for the element, `None` is returned.

    See Also
    --------
    vdw_radius_protor : *Van-der-Waals* radii for structures without annotated hydrogen atoms.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> print(vdw_radius_single("C"))
    1.7
    """
    return _SINGLE_ATOM_VDW_RADII.get(element.upper())
