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
from ...atoms import AtomArray, AtomArrayStack
from ...charges import partial_charges
from ...bonds import BondList, BondType, find_connected, find_rotatable_bonds


PARAMETRIZED_ELEMENTS = [
    "H", "C", "N", "O", "P", "S",
    "F", "CL", "BR", "I",
    "MG", "CA", "MN", "FE", "ZN"
]


class PDBQTFile(TextFile):

    def get_remarks(self, model=None):
        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self.lines))
                                  if self.lines[i].startswith(("MODEL"))],
                                 dtype=int)
        # Line indices with ATOM or HETATM records
        remark_line_i = np.array([i for i in range(len(self.lines)) if
                                  self.lines[i].startswith("REMARK")],
                                 dtype=int)
        # Structures containing only one model may omit MODEL record
        # In these cases model starting index is set to 0
        if len(model_start_i) == 0:
            model_start_i = np.array([0])
        
        if model is None:
            # Add exclusive end of file
            model_start_i = np.concatenate((model_start_i, [len(self.lines)]))
            model_i = 0
            remarks = []
            for i in range(len(model_start_i) - 1):
                start = model_start_i[i]
                stop  = model_start_i[i+1]
                model_remark_line_i = remark_line_i[
                    (remark_line_i >= start) & (remark_line_i < stop)
                ]
                remarks.append(
                    "\n".join([self.lines[i][7:] for i in model_remark_line_i])
                )
            return remarks
        
        else:
            last_model = len(model_start_i)
            if model == 0:
                raise ValueError("The model index must not be 0")
            # Negative models mean index starting from last model
            model = last_model + model + 1 if model < 0 else model

            if model < last_model:
                line_filter = ( ( remark_line_i >= model_start_i[model-1] ) &
                                ( remark_line_i <  model_start_i[model  ] ) )
            elif model == last_model:
                line_filter = (remark_line_i >= model_start_i[model-1])
            else:
                raise ValueError(
                    f"The file has {last_model} models, "
                    f"the given model {model} does not exist"
                )
            remark_line_i = remark_line_i[line_filter]
            
            # Do not include 'REMARK ' itself -> begin from pos 8
            return "\n".join([self.lines[i][7:] for i in remark_line_i])


    def get_structure(self, model=None):
        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self.lines))
                                  if self.lines[i].startswith(("MODEL"))],
                                 dtype=int)
        # Line indices with ATOM or HETATM records
        atom_line_i = np.array([i for i in range(len(self.lines)) if
                                self.lines[i].startswith(("ATOM", "HETATM"))],
                               dtype=int)
        # Structures containing only one model may omit MODEL record
        # In these cases model starting index is set to 0
        if len(model_start_i) == 0:
            model_start_i = np.array([0])
        
        if model is None:
            depth = len(model_start_i)
            length = self._get_model_length(model_start_i, atom_line_i)
            array = AtomArrayStack(depth, length)
            # Line indices for annotation determination
            # Annotation is determined from model 1,
            # therefore from ATOM records before second MODEL record
            if len(model_start_i) == 1:
                annot_i = atom_line_i
            else:
                annot_i = atom_line_i[atom_line_i < model_start_i[1]]
            # Line indices for coordinate determination
            coord_i = atom_line_i
        
        else:
            last_model = len(model_start_i)
            if model == 0:
                raise ValueError("The model index must not be 0")
            # Negative models mean index starting from last model
            model = last_model + model + 1 if model < 0 else model

            if model < last_model:
                line_filter = ( ( atom_line_i >= model_start_i[model-1] ) &
                                ( atom_line_i <  model_start_i[model  ] ) )
            elif model == last_model:
                line_filter = (atom_line_i >= model_start_i[model-1])
            else:
                raise ValueError(
                    f"The file has {last_model} models, "
                    f"the given model {model} does not exist"
                )
            annot_i = coord_i = atom_line_i[line_filter]
            array = AtomArray(len(coord_i))
        
        # Save atom IDs for later sorting into the original atom order
        atom_id  = np.zeros(array.array_length(), int)

        # Create annotation arrays
        chain_id  = np.zeros(array.array_length(), array.chain_id.dtype)
        res_id    = np.zeros(array.array_length(), array.res_id.dtype)
        ins_code  = np.zeros(array.array_length(), array.ins_code.dtype)
        res_name  = np.zeros(array.array_length(), array.res_name.dtype)
        hetero    = np.zeros(array.array_length(), array.hetero.dtype)
        atom_name = np.zeros(array.array_length(), array.atom_name.dtype)
        element   = np.zeros(array.array_length(), array.element.dtype)

        # Fill annotation array
        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            
            atom_id[i] = int(line[6:11])
            chain_id[i] = line[21].upper().strip()
            res_id[i] = int(line[22:26])
            ins_code[i] = line[26].strip()
            res_name[i] = line[17:20].strip()
            hetero[i] = (False if line[0:4] == "ATOM" else True)
            atom_name[i] = line[12:16].strip()
            element[i] = line[76:78].strip()
        
        # Add annotation arrays to atom array (stack)
        array.chain_id = chain_id
        array.res_id = res_id
        array.ins_code = ins_code
        array.res_name = res_name
        array.hetero = hetero
        array.atom_name = atom_name
        array.element = element
        
        # Fill in coordinates
        if isinstance(array, AtomArray):
            for i, line_i in enumerate(coord_i):
                line = self.lines[line_i]
                array.coord[i,0] = float(line[30:38])
                array.coord[i,1] = float(line[38:46])
                array.coord[i,2] = float(line[46:54])
                
        elif isinstance(array, AtomArrayStack):
            m = 0
            i = 0
            for line_i in atom_line_i:
                if m < len(model_start_i)-1 and line_i > model_start_i[m+1]:
                    m += 1
                    i = 0
                line = self.lines[line_i]
                array.coord[m,i,0] = float(line[30:38])
                array.coord[m,i,1] = float(line[38:46])
                array.coord[m,i,2] = float(line[46:54])
                i += 1
        
        # Sort into the original atom order
        array = array[..., np.argsort(atom_id)]

        return array
    

    def set_structure(self, atoms, charges=None, atom_types=None,
                      rotatable_bonds=None, rigid_root=None,
                      include_torsdof=True):
        if charges is None:
            charges = partial_charges(atoms)
            charges[np.isnan(charges)] = 0
        else:
            if np.isnan(charges).any():
                raise ValueError("Input charges contain NaN values")
        
        # Get AutoDock atom types and remove unpolar hydrogen atoms
        atoms, charges, types, mask = convert_atoms(atoms, charges)
        # Overwrite calculated atom types with input atom types
        if atom_types is not None:
            types = atom_types[mask]
        
        if rotatable_bonds is None:
            # No rotatable bonds -> the BondList contains no bonds
            rotatable_bonds = BondList(atoms.bonds.get_atom_count())
            use_root = False
        elif rotatable_bonds == "rigid":
            rotatable_bonds = BondList(atoms.bonds.get_atom_count())
            use_root = True
        elif rotatable_bonds == "all":
            rotatable_bonds = find_rotatable_bonds(atoms.bonds)
            use_root = True
        else:
            if rotatable_bonds.ndim != 2 or rotatable_bonds.shape[1] != 2:
                raise ValueError(
                    "An (nx2) array is expected for rotatable bonds"
                )
            rotatable_bonds = BondList(
                len(mask), np.asarray(rotatable_bonds)
            )[mask]
            use_root = True
        
        if rigid_root is None:
            root_index = 0
        else:
            # Find new index of root atom, since the index might have
            # been shifted due to removed atoms
            original_indices = np.arange(len(mask))
            new_indices = original_indices[mask]
            try:
                root_index = np.where(new_indices == rigid_root)[0][0]
            except IndexError:
                raise ValueError(
                    "The given root atom index points to an unpolar hydrogen "
                    "atom, that has been removed"
                )
            # Add bonds of the rigid root to rotatable bonds,
            # as they probably have been filtered out,
            # as the root is probably a terminal atom
            for atom, bond_type in zip(*atoms.bonds.get_bonds(root_index)):
                rotatable_bonds.add_bond(root_index, atom, bond_type)
        
        # Break rotatable bonds
        # for simple branch determination in '_write_atoms()'
        atoms.bonds.remove_bonds(rotatable_bonds)

        hetero = ["ATOM" if e == False else "HETATM" for e in atoms.hetero]
        if "atom_id" in atoms.get_annotation_categories():
            atom_id = atoms.atom_id
        else:
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
            root_index, rotatable_bonds,
            np.zeros(len(rotatable_bonds), dtype=bool), use_root
        )
        if include_torsdof:
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
                f"{atoms.res_id[i]:>4d}"
                f"{atoms.ins_code[i]:1}   "
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
    

    def _get_model_length(self, model_start_i, atom_line_i):
        """
        Determine length of models and check that all models
        have equal length.
        """
        n_models = len(model_start_i)
        length = None
        for model_i in range(len(model_start_i)):
            model_start = model_start_i[model_i]
            model_stop = model_start_i[model_i+1] if model_i+1 < n_models \
                            else len(self.lines)
            model_length = np.count_nonzero(
                (atom_line_i >= model_start) & (atom_line_i < model_stop)
            )
            if length is None:
                length = model_length
            if model_length != length:
                raise InvalidFileError(
                    f"Model {model_i+1} has {model_length} atoms, "
                    f"but model 1 has {length} atoms, must be equal"
                )
        return length


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