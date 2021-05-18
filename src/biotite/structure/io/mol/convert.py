# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["get_structure", "set_structure"]



def get_structure(mol_file):
    return mol_file.get_structure()
    
def set_structure(mol_file, atoms):
    mol_file.set_structure(atoms)