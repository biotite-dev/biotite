# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["vdw_radius_protor", "vdw_radius_single"]

from .bonds import bonds_in_residue


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
    ("F",  1, 0) : 1.47, # Taken from _SINGLE_RADII
    ("CL", 1, 0) : 1.75, # Taken from _SINGLE_RADII
    ("BR", 1, 0) : 1.85, # Taken from _SINGLE_RADII
    ("I",  1, 0) : 1.98, # Taken from _SINGLE_RADII
}

_SINGLE_RADII = {
    "H":  1.20,
    "HE": 1.40,
    
    "C":  1.70,
    "N":  1.55,
    "O":  1.52,
    "F":  1.47,
    "NE": 1.54,
    
    "SI": 2.10,
    "P":  1.80,
    "S":  1.80,
    "CL": 1.75,
    "AR": 1.88,
    
    "AS": 1.85,
    "SE": 1.90,
    "BR": 1.85,
    "KR": 2.02,
    
    "TE": 2.06,
    "I":  1.98,
    "XE": 2.16,
}

# A dictionary that caches radii for each residue
_protor_radii = {}


def vdw_radius_protor(res_name, atom_name):
    """
    Estimate the Van-der-Waals radius of an non-hydrogen atom,
    that includes the radius added by potential bonded hydrogen atoms.
    The respective radii are taken from the ProtOr dataset. [1]_.

    This is especially useful for macromolecular structures where no
    hydrogen atoms are resolved, e.g. crystal structures.
    The valency of the non-hydrogen atom and the amount of normally
    bonded hydrogen atoms is taken from the chemical compound dictionary
    dataset.

    Parameters
    ----------
    res_name : str
        The up to 3-letter residue name the non-hydrogen atom belongs
        to.
    atom_name : str
        The name of the non-hydrogen atom.
    
    Returns
    -------
    The Van-der-Waals radius of the given atom.
    If the radius cannot be estimated for the atom, `None` is returned.

    See also
    --------
    vdw_radius_single
    
    References
    ----------
   
    .. [1] J Tsai R Taylor, C Chotia and M Gerstein,
       "The packing densitiy in proteins: standard radii and volumes."
       J Mol Biol, 290, 253-299 (1999).
    
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
                f"Residue '{res_name}' does not contain an atom named "
                f"'{atom_name}'"
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
    radii = {atom : _PROTOR_RADII.get(tuple(group))
             for atom, group in groups.items()}
    return radii


def vdw_radius_single(element):
    """
    Get the Van-der-Waals radius of an atom from the given element. [1]_

    Parameters
    ----------
    element : str
        The chemical element of the atoms.
    
    Returns
    -------
    The Van-der-Waals radius of the atom.
    If the radius is unknown for the element, `None` is returned.
    
    See also
    --------
    vdw_radius_protor
    
    References
    ----------
       
    .. [1] A Bondi,
       "Van der Waals volumes and radii."
       J Phys Chem, 86, 441-451 (1964).
    
    Examples
    --------

    >>> print(vdw_radius_single("C"))
    1.7
    """
    return _SINGLE_RADII.get(element.upper())