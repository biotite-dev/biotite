# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbqt"
__author__ = "Patrick Kunzmann, Daniel Bauer"
__all__ = ["PDBQTFile"]

import warnings
import numpy as np
import networkx as nx
from ....file import TextFile, InvalidFileError
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
    """
    This class represents an *AutoDock* PDBQT file.
    
    This class only provides rudimentary support for reading/writing
    the pure atom information.

    EXPERIMENTAL: Future API changes are probable.
    
    Examples
    --------
    
    Write biotin as flexible ligand into a PDBQT file:
    
    >>> import os.path
    >>> ligand = residue("BTN")
    >>> file = PDBQTFile()
    >>> mask = file.set_structure(ligand, rotatable_bonds="all")
    >>> # Print removed nonpolar hydrogen atoms
    >>> print(ligand[~mask])
    HET         0  BTN H101   H         3.745    1.171    0.974
    HET         0  BTN H102   H         4.071    1.343   -0.767
    HET         0  BTN H91    H         2.802   -0.740   -1.211
    HET         0  BTN H92    H         2.476   -0.912    0.530
    HET         0  BTN H81    H         1.289    1.265    0.523
    HET         0  BTN H82    H         1.616    1.437   -1.218
    HET         0  BTN H71    H         0.346   -0.646   -1.662
    HET         0  BTN H72    H         0.020   -0.818    0.079
    HET         0  BTN H2     H        -0.838    1.576   -1.627
    HET         0  BTN H61    H        -3.797    1.837    1.286
    HET         0  BTN H62    H        -3.367    2.738   -0.205
    HET         0  BTN H5     H        -4.307    0.812   -1.205
    HET         0  BTN H4     H        -2.451   -0.038   -2.252
    >>> print(file)
    ROOT
    HETATM    1 C11  BTN     0       5.089  -0.280   0.173  1.00  0.00     0.258 C 
    HETATM    2 O11  BTN     0       4.956  -1.473   0.030  1.00  0.00    -0.264 OA
    ENDROOT
    BRANCH   1   3
    HETATM    3 O12  BTN     0       6.299   0.233   0.444  1.00  0.00    -0.331 OA
    HETATM   17 HO2  BTN     0       7.034  -0.391   0.517  1.00  0.00     0.221 HD
    ENDBRANCH   1   3
    BRANCH   1   4
    HETATM    4 C10  BTN     0       3.896   0.631   0.039  1.00  0.00     0.105 C 
    BRANCH   4   5
    HETATM    5 C9   BTN     0       2.651  -0.200  -0.276  1.00  0.00     0.010 C 
    BRANCH   5   6
    HETATM    6 C8   BTN     0       1.440   0.725  -0.412  1.00  0.00     0.002 C 
    BRANCH   6   7
    HETATM    7 C7   BTN     0       0.196  -0.106  -0.727  1.00  0.00     0.016 C 
    BRANCH   7   8
    HETATM    8 C2   BTN     0      -1.015   0.819  -0.863  1.00  0.00     0.065 C 
    HETATM    9 S1   BTN     0      -1.419   1.604   0.751  1.00  0.00    -0.154 SA
    HETATM   10 C6   BTN     0      -3.205   1.827   0.371  1.00  0.00     0.090 C 
    HETATM   11 C5   BTN     0      -3.530   0.581  -0.476  1.00  0.00     0.091 C 
    HETATM   12 N1   BTN     0      -3.970  -0.507   0.412  1.00  0.00    -0.239 NA
    HETATM   13 C3   BTN     0      -3.141  -1.549   0.271  1.00  0.00     0.272 C 
    HETATM   14 O3   BTN     0      -3.271  -2.589   0.888  1.00  0.00    -0.259 OA
    HETATM   15 N2   BTN     0      -2.154  -1.343  -0.612  1.00  0.00    -0.239 NA
    HETATM   16 C4   BTN     0      -2.289   0.010  -1.175  1.00  0.00     0.093 C 
    HETATM   18 HN1  BTN     0      -4.738  -0.474   1.004  1.00  0.00     0.132 HD
    HETATM   19 HN2  BTN     0      -1.462  -1.982  -0.843  1.00  0.00     0.132 HD
    ENDBRANCH   7   8
    ENDBRANCH   6   7
    ENDBRANCH   5   6
    ENDBRANCH   4   5
    ENDBRANCH   1   4
    TORSDOF 6
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.pdb"))
    """

    def get_remarks(self, model=None):
        """
        Get the content of ``REMARKS`` lines.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return a
            string from the remarks corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model insted of the first model.
            If this parameter is omitted, a list of strings
            containing all models will be returned, even if the
            structure contains only one model.

        Returns
        -------
        lines : str or list of str
            The content of ``REMARKS`` lines, without the leading
            ``'REMARKS'``.
        """
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
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the
        PDBQT file.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model insted of the first model.
            If this parameter is omitted, an :class:`AtomArrayStack`
            containing all models will be returned, even if the
            structure contains only one model.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """
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
                      rotatable_bonds=None, root=None, include_torsdof=True):
        """
        Write an :class:`AtomArray` into the PDBQT file.
        
        Parameters
        ----------
        atoms : AtomArray, shape=(n,)
            The atoms to be written into this file.
            Must have an associated :class:`BondList`.
        charges : ndarray, shape=(n,), dtype=float, optional
            Partial charges for each atom in `atoms`.
            By default, the charges are calculated using the PEOE method
            (:func:`partial_charges()`).
        atom_types : ndarray, shape=(n,), dtype="U1", optional
            Custom *AutoDock* atom types for each atom in `atoms`.
        rotatable_bonds : None or 'rigid' or 'all' or BondList, optional
            This parameter describes, how rotatable bonds are handled,
            with respect to ``ROOT``, ``BRANCH`` and ``ENDBRANCH``
            lines.

                - ``None`` - The molecule is handled as rigid receptor:
                  No ``ROOT``, ``BRANCH`` and ``ENDBRANCH`` lines will
                  be written.
                - ``'rigid'`` - The molecule is handled as rigid ligand:
                  Only a ``ROOT`` line will be written.
                - ``'all'`` - The molecule is handled as flexible 
                  ligand:
                  A ``ROOT`` line will be written and all rotatable
                  bonds are included using ``BRANCH`` and ``ENDBRANCH``
                  lines.
                - :class:`BondList` - The molecule is handled as
                  flexible ligand:
                  A ``ROOT`` line will be written and all bonds in the
                  given :class:`BondList` are considered flexible via
                  ``BRANCH`` and ``ENDBRANCH`` lines.
            
        root : int, optional
            Specifies the index of the atom following the ``ROOT`` line.
            Setting the root atom is useful for specifying the *anchor*
            in flexible side chains.
            This parameter has no effect, if `rotatable_bonds` is
            ``None``.
            By default, the first atom is also the root atom.
        include_torsdof : bool, optional
            By default, a ``TORSDOF`` (torsional degrees of freedom)
            record is written at the end of the file.
            By setting this parameter to false, the record is omitted.
        
        Returns
        -------
        mask : ndarray, shape=(n,), dtype=bool
            A boolean mask, that is ``False`` for each atom of the input
            ``atoms``, that was removed due to being a nonpolar
            hydrogen.
        """
        if charges is None:
            charges = partial_charges(atoms)
            charges[np.isnan(charges)] = 0
        else:
            if np.isnan(charges).any():
                raise ValueError("Input charges contain NaN values")
        
        # Get AutoDock atom types and remove nonpolar hydrogen atoms
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
        
        if root is None:
            root_index = 0
        else:
            # Find new index of root atom, since the index might have
            # been shifted due to removed atoms
            original_indices = np.arange(len(mask))
            new_indices = original_indices[mask]
            try:
                root_index = np.where(new_indices == root)[0][0]
            except IndexError:
                raise ValueError(
                    "The given root atom index points to an nonpolar hydrogen "
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
    """
    Convert atoms into *AutoDock* compatible atoms.

    Parameters
    ----------
    atoms : AtomArray
        The atoms to be converted.
    charges : ndarray, dtype=float
        Partial charges for the atoms.
    
    Returns
    -------
    converted_atoms : AtomArray
        The input `atoms`, but with deleted nonpolar hydrogen atoms.
    charges : ndarray, dtype=float
        The input `charges`, but with deleted entries for nonpolar
        hydrogen atoms.
    atom_types : ndarray, dtype="U1"
        The *AutoDock* atom types.
    mask : ndarray, shape=(n,), dtype=bool
        A boolean mask, that is ``False`` for each atom of the input
        ``atoms``, that was removed due to being a nonpolar hydrogen.
    """
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
            if np.isin(
                all_bond_types[i],
                [BondType.AROMATIC_SINGLE, BondType.AROMATIC_DOUBLE]
            ).any():
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