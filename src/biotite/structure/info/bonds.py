# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["get_bond_dataset", "get_bond_order", "get_bonds_for_residue"]

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

    _info_dir = dirname(realpath(__file__))
    with open(join(_info_dir, "intra_bonds.msgpack"), "rb") as file:
        _intra_bonds_raw = msgpack.unpack(
            file, use_list=False, raw=False
        )
        _intra_bonds = {}
        for group, group_bonds_raw in _intra_bonds_raw.items():
            group_bonds = {
                frozenset(bond_raw) : count
                for bond_raw, count in group_bonds_raw.items()
            }
            _intra_bonds[group] = group_bonds


def get_bond_dataset():
    _init_dataset()
    return copy.copy(_intra_bonds)


def get_bond_order(res_name, atom_name1, atom_name2):
    _init_dataset()
    group_bonds = _intra_bonds.get(res_name)
    if group_bonds is None:
        return None
    else:
        return group_bonds.get(frozenset((atom_name1, atom_name2)))


def get_bonds_for_residue(res_name):
    _init_dataset()
    return copy.copy(_intra_bonds.get(res_name))