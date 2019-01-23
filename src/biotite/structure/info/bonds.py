# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["get_bond_database", "get_bond_order", "get_bonds_for_residue"]

import json
import copy
from os.path import join, dirname, realpath

_info_dir = dirname(realpath(__file__))
with open(join(_info_dir, "intra_bonds.json")) as file:
    _intra_bonds_raw = json.load(file)
    _intra_bonds = {}
    for group, group_bonds_raw in _intra_bonds_raw.items():
        group_bonds = {}
        for bond_raw, count in group_bonds_raw.items():
            group_bonds[frozenset(bond_raw.split())] = count
        _intra_bonds[group] = group_bonds


def get_bond_database():
    return copy.copy(_intra_bonds)


def get_bond_order(res_name, atom_name1, atom_name2):
    group_bonds = _intra_bonds.get(res_name)
    if group_bonds is None:
        return None
    else:
        return group_bonds.get(frozenset(atom_name1, atom_name2))


def get_bonds_for_residue(res_name):
    return copy.copy(_intra_bonds.get(res_name))