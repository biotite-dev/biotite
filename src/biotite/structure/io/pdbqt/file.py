# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann, Daniel Bauer"
__all__ = ["PDBQTFile"]

import warnings
import numpy as np
import networkx as nx
from ....file import TextFile
from ...error import BadStructureError
from ...charges import partial_charges
from ...bonds import BondList, BondType, find_connected


PARAMETRIZED_ELEMENTS = [
    "H", "C", "N", "O", "P", "S",
    "F", "CL", "BR", "I",
    "MG", "CA", "MN", "FE", "ZN"
]


class PDBQTFile(TextFile):

    def get_structure(self, model=None, include_charge=False):
        pass


    def set_structure(self, atoms, charges=None, atom_types=None,
                      rotatable_bonds=None):
        if charges is None:
            charges = partial_charges(atoms)
        
        atoms, charges, types, mask = convert_atoms(atoms, charges)
        if atom_types is not None:
            types = atom_types[mask]
        
        if rotatable_bonds is None:
            rotatable_bonds = BondList(atoms.bonds.get_atom_count())
        elif rotatable_bonds == "all":
            rotatable_bonds = find_rotatable_bonds(atoms.bonds)
        else:
            rotatable_bonds = BondList(
                atoms.bonds.get_atom_count(), np.asarray(rotatable_bonds)
            )
            if rotatable_bonds.ndim != 2 or rotatable_bonds.shape[1] != 2:
                raise ValueError(
                    "An (nx2) array is expected for rotatable bonds"
                )
        # Break rotatable bonds
        # for simple branch determination in '_write_atoms()'
        atoms.bonds.remove_bonds(rotatable_bonds)

        hetero = ["ATOM" if e == False else "HETATM" for e in atoms.hetero]
        atom_id = np.arange(1, atoms.array_length()+1)
        occupancy = np.ones(atoms.array_length())
        b_factor = np.zeros(atoms.array_length())


        # Convert rotatable bonds into array for easier handling
        # The bond type is irrelevant from this point on
        rotatable_bonds = rotatable_bonds.as_array()[:,:2]

        self.lines = []
        self._write_atoms(
            atoms, charges, types,
            atom_id, hetero, occupancy, b_factor,
            0, rotatable_bonds, np.zeros(len(rotatable_bonds), dtype=bool),
            True
        )
        self.lines.append(f"TORSDOF {len(rotatable_bonds)}")

        return mask
    

    def _write_atoms(self, atoms, charges, types,
                     atom_id, hetero, occupancy, b_factor,
                     root_atom, rotatable_bonds, visited_rotatable_bonds,
                     is_root):
        if len(rotatable_bonds) != 0:
            # Get the indices to atoms of this branch, i.e. a group of
            # atoms that are connected by non-rotatable bonds
            # Use 'find_connected()', since rotatable bonds were removed
            # from the BondList before
            this_branch_indices = find_connected(atoms.bonds, root_atom)
            # The root atom of the branch, i.e. the atom connected by
            # the rotatable bond should always be listed first
            # -> Remove root atom and insert it at the beginning
            this_branch_indices = np.insert(
                this_branch_indices[this_branch_indices != root_atom],
                0,
                root_atom
            )
        else:
            # No rotatable bonds
            # -> all atom are in root i.e. this branch
            this_branch_indices = np.arange(atoms.array_length())
        
        if is_root:
            self.lines.append("ROOT")
        for i in this_branch_indices:
            self.lines.append(
                f"{hetero[i]:6}"
                f"{atom_id[i]:>5d} "
                f"{atoms.atom_name[i]:4} "
                f"{atoms.res_name[i]:3} "
                f"{atoms.chain_id[i]:1}"
                f"{atoms.res_id[i]:>4d}    "
                f"{atoms.coord[i,0]:>8.3f}"
                f"{atoms.coord[i,1]:>8.3f}"
                f"{atoms.coord[i,2]:>8.3f}"
                f"{occupancy[i]:>6.2f}"
                f"{b_factor[i]:>6.2f}    "
                f"{charges[i]:>6.3f} "
                f"{types[i]:2}"
            )
        if is_root:
            self.lines.append("ENDROOT")

        if len(rotatable_bonds) == 0:
            # No rotatable bonds -> no branching
            return

        for k, (i, j) in enumerate(rotatable_bonds):
            if visited_rotatable_bonds[k]:
                continue

            # Create a new branch for each rotatable bond,
            # that connects to an atom of this branch 
            if i in this_branch_indices:
                this_br_i = i
                new_br_i = j
            elif j in this_branch_indices:
                this_br_i = j
                new_br_i = i
            else:
                # Rotatable bond does not start from this branch
                continue
            
            # Mark rotatable bond as visited as otherwise branches would
            # be created back and forth over the same rotatable bond and
            # this method would never terminate
            visited_rotatable_bonds[k] = True

            self.lines.append(
                f"BRANCH {atom_id[this_br_i]:>3d} {atom_id[new_br_i]:>3d}"
            )
            self._write_atoms(
                atoms, charges, types,
                atom_id, hetero, occupancy, b_factor,
                # The root atom of the branch
                #is the other atom of the rotatable bond
                new_br_i, rotatable_bonds, visited_rotatable_bonds,
                False
            )
            self.lines.append(
                f"ENDBRANCH {atom_id[this_br_i]:>3d} {atom_id[new_br_i]:>3d}"
            )


def convert_atoms(atoms, charges):
    charges = charges.copy()
    all_bonds, all_bond_types = atoms.bonds.get_all_bonds()

    atom_types = np.zeros(atoms.array_length(), dtype="U2")
    hydrogen_removal_mask = np.zeros(atoms.array_length(), dtype=bool)
    for i in range(atoms.array_length()):
        element = atoms.element[i]
        bonded_atoms = all_bonds[i][all_bonds[i] != -1]
        if element == "H":
            if len(bonded_atoms) == 0:
                # Free proton
                atom_types[i] = "H"
            elif len(bonded_atoms) == 1:
                j = bonded_atoms[0]
                bonded_element = atoms.element[j]
                if bonded_element == "C":
                    # Remove hydrogen and add its charge
                    # to charge of bonded carbon
                    charges[j] += charges[i]
                    hydrogen_removal_mask[i] = True
                else:
                    atom_types[i] = "HD"
            else:
                raise BadStructureError(
                    "Structure contains hydrogen with multiple bonds"
                )
        elif element == "C":
            if (all_bond_types[i] == BondType.AROMATIC).any():
                # Aromatic carbon
                atom_types[i] = "A"
            else:
                # Alphatic carbon
                atom_types[i] = "C"
        elif element == "N":
            atom_types[i] = "NA"
        elif element == "O":
            atom_types[i] = "OA"
        elif element == "S":
            atom_types[i] = "SA"
        elif element in PARAMETRIZED_ELEMENTS:
            atom_types[i] = element
        else:
            warnings.warn(
                f"Element {element} is not paramtrized, "
                f"using parameters for hydrogen instead"
            ) 
            atom_types[i] = "H"
    
    mask = ~hydrogen_removal_mask
    return atoms[mask], charges[mask], atom_types[mask], mask


def find_rotatable_bonds(bonds):
    bond_graph = bonds.as_graph()
    cycles = nx.algorithms.cycles.cycle_basis(bond_graph)

    number_of_partners = np.count_nonzero(
        bonds.get_all_bonds()[0] != -1,
        axis=1
    )

    rotatable_bonds = []
    for i, j, bond_type in bonds.as_array():
        # Can only rotate about single bonds
        # Furthermore, it makes no sense to rotate about a bond,
        # that leads to a single atom
        if bond_type == BondType.SINGLE \
            and number_of_partners[i] > 1 \
            and number_of_partners[j] > 1:
                # Cannot rotate about a bond, if the two connected atoms
                # are in a cycle
                in_same_cycle = False
                for cycle in cycles:
                    if i in cycle and j in cycle:
                        in_same_cycle = True
                if not in_same_cycle:
                    rotatable_bonds.append((i,j, bond_type))
    return BondList(bonds.get_atom_count(), np.array(rotatable_bonds))