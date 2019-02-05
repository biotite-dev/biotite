# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["vdw_radius_protor", "vdw_radius_single"]

from .bonds import get_bonds_for_residue


_PROTOR_DEFAULT = 1.80
# Contains tuples for the different ProtOr groups:
# Element, valency, H count
_PROTOR_RADII = {
    ("C", 3, 0) : 1.61,
    ("C", 3, 1) : 1.76,
    ("C", 4, 1) : 1.88,
    ("C", 4, 2) : 1.88,
    ("C", 4, 3) : 1.88,
    ("N", 3, 0) : 1.64,
    ("N", 3, 1) : 1.64,
    ("N", 3, 2) : 1.64,
    ("N", 4, 3) : 1.64,
    ("N", 4, 4) : 1.64, # Not official, added for completeness
    ("O", 1, 0) : 1.42,
    ("O", 2, 0) : 1.46, # Not official, added for completeness
    ("O", 2, 1) : 1.46,
    ("S", 1, 0) : 1.77,
    ("S", 2, 0) : 1.77, # Not official, added for completeness
    ("S", 2, 1) : 1.77,
    ("S", 2, 2) : 1.77  # Not official, added for completeness
}

_SINGLE_DEFAULT = 1.50
_SINGLE_RADII = {
    "H":  1.20,
    "C":  1.70,
    "N":  1.55,
    "O":  1.52,
    "F":  1.47,
    "SI": 2.10,
    "P":  1.80,
    "S":  1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I":  1.98
}

# A precalculated dictionary of radii for each residue
_protor_radii = {}


def vdw_radius_protor(res_name, atom_name):
    res_name = res_name.upper()
    if res_name in _protor_radii:
        return _protor_radii[res_name][atom_name]
    else:
        _protor_radii[res_name] = _calculate_protor_radii(res_name)
        return _protor_radii[res_name][atom_name]

def _calculate_protor_radii(res_name):
    bonds = get_bonds_for_residue(res_name)
    # Element, valency, H count
    groups = {}
    for atom1, atom2 in bonds:
        for main_atom, bound_atom in ((atom1, atom2), (atom2, atom1)):
            if main_atom[0] not in ["C", "N", "O", "S"]:
                continue
            group = groups.get(main_atom)
            if group is None:
                group = [main_atom[0], 0, 0]
            group[1] += 1
            if bound_atom[0] == "H":
                group[2] += 1
            groups[main_atom] = group
    radii = {atom : _PROTOR_RADII.get(tuple(group), _PROTOR_DEFAULT)
             for atom, group in groups.items()}
    return radii


def vdw_radius_single(element):
    return _SINGLE_RADII.get(element.upper(), _SINGLE_DEFAULT)