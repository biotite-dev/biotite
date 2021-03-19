# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Some convenience functions for consistency with other ``structure.io``
subpackages.
"""

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]


def get_structure(pdbqt_file, model=None):
    return pdbqt_file.get_structure(model)


def set_structure(pdbqt_file, atoms, charges=None, atom_types=None,
                  rotatable_bonds=None, rigid_root=None, include_torsdof=True):
    return pdbqt_file.set_structure(
        atoms, charges, atom_types, rotatable_bonds, rigid_root,
        include_torsdof
    )