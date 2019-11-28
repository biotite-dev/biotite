# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann, Daniel Bauer"
__all__ = ["PDBFile"]

import numpy as np
from ...atoms import AtomArray, AtomArrayStack
from ...box import vectors_from_unitcell, unitcell_from_vectors
from ....file import TextFile, InvalidFileError
from ...error import BadStructureError
from ...filter import filter_altloc
from .hybrid36 import encode_hybrid36, decode_hybrid36, max_hybrid36_number
import copy
from warnings import warn


_atom_records = {"hetero"    : (0,  6),
                 "atom_id"   : (6,  11),
                 "atom_name" : (12, 16),
                 "alt_loc"   : (16, ),
                 "res_name"  : (17, 20),
                 "chain_id"  : (21, ),
                 "res_id"    : (22, 26),
                 "ins_code"  : (26, ),
                 "coord_x"   : (30, 38),
                 "coord_y"   : (38, 46),
                 "coord_z"   : (46, 54),
                 "occupancy" : (54, 60),
                 "temp_f"    : (60, 66),
                 "element"   : (76, 78),
                 "charge"    : (78, 80),}


class PDBFile(TextFile):
    r"""
    This class represents a PDB file.
    
    The usage of PDBxFile is encouraged in favor of this class.
    
    This class only provides support for reading/writing the pure atom
    information (*ATOM*, *HETATM*, *MODEL* and *ENDMDL* records). *TER*
    records cannot be written.
    
    See also
    --------
    PDBxFile
    
    Examples
    --------
    Load a `\\*.pdb` file, modify the structure and save the new
    structure into a new file:
    
    >>> import os.path
    >>> file = PDBFile()
    >>> file.read(os.path.join(path_to_structures, "1l2y.pdb"))
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = PDBFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.pdb"))
    """
    
    def get_coord(self, model=None):
        """
        Get only the coordinates of the PDB file.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return a
            2D coordinate array from the atoms corresponding to the
            given model number.
            If this parameter is omitted, an 2D coordinate array
            containing all models will be returned, even if
            the structure contains only one model.
        
        Returns
        -------
        coord : ndarray, shape=(m,n,3) or shape=(n,2), dtype=float
            The coordinates read from the ATOM and HETATM records of the
            file.
        
        Notes
        -----
        Note that :func:`get_coord()` may output more coordinates than
        the atom array (stack) from the corresponding
        :func:`get_structure()` call has.
        The reason for this is, that :func:`get_structure()` filters
        *altlocs*, while `get_coord()` does not.
        
        Examples
        --------
        Read an :class:`AtomArrayStack` from multiple PDB files, where
        each PDB file contains the same atoms but different positions.
        This is an efficient approach when a trajectory is spread into
        multiple PDB files, as done e.g. by the *Rosetta* modeling
        software. 

        For the purpose of this example, the PDB files are created from
        an existing :class:`AtomArrayStack`.
        
        >>> import os.path
        >>> file_names = []
        >>> for i in range(atom_array_stack.stack_depth()):
        ...     pdb_file = PDBFile()
        ...     pdb_file.set_structure(atom_array_stack[i])
        ...     file_name = os.path.join(temp_dir(), f"model_{i+1}.pdb")
        ...     pdb_file.write(file_name)
        ...     file_names.append(file_name)
        >>> print(file_names)
        ['...model_1.pdb', '...model_2.pdb', ..., '...model_38.pdb']

        Now the PDB files are used to create an :class:`AtomArrayStack`,
        where each model represents a different model.

        Construct a new :class:`AtomArrayStack` with annotations taken
        from one of the created files used as template.

        >>> template_file = PDBFile()
        >>> template_file.read(file_names[0])
        >>> template_array = template_file.get_structure(model=1)
        >>> new_stack = AtomArrayStack(len(file_names), template_array.array_length())
        >>> for category in template_array.get_annotation_categories():
        ...     annot = template_array.get_annotation(category)
        ...     new_stack.set_annotation(category, annot)

        Fill coordinates of new :class:`AtomArrayStack` with coordinates
        of the PDB files.

        >>> for i, file_name in enumerate(file_names):
        ...     pdb_file = PDBFile()
        ...     pdb_file.read(file_name)
        ...     new_stack.coord[i] = pdb_file.get_coord(model=1)

        The newly created :class:`AtomArrayStack` should now be equal to
        the :class:`AtomArrayStack` the PDB files were created from.

        >>> print(new_stack == atom_array_stack)
        True
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
            coord_i = atom_line_i
        
        else:
            last_model = len(model_start_i)
            if model < last_model:
                line_filter = ( ( atom_line_i >= model_start_i[model-1] ) &
                                ( atom_line_i <  model_start_i[model  ] ) )
            elif model == last_model:
                line_filter = (atom_line_i >= model_start_i[model-1])
            else:
                raise ValueError(
                    f"Requested model number {model} is larger than the "
                    f"amount of models ({last_model})"
                )
            coord_i = atom_line_i[line_filter]
            length = len(coord_i)
        
        # Fill in coordinates
        if model is None:
            coord = np.zeros((depth, length, 3), dtype=np.float32)
            m = 0
            i = 0
            for line_i in atom_line_i:
                if m < len(model_start_i)-1 and line_i > model_start_i[m+1]:
                    m += 1
                    i = 0
                line = self.lines[line_i]
                coord[m,i,0] = float(line[30:38])
                coord[m,i,1] = float(line[38:46])
                coord[m,i,2] = float(line[46:54])
                i += 1
            return coord
        
        else:
            coord = np.zeros((length, 3), dtype=np.float32)
            for i, line_i in enumerate(coord_i):
                line = self.lines[line_i]
                coord[i,0] = float(line[30:38])
                coord[i,1] = float(line[38:46])
                coord[i,2] = float(line[46:54])
            return coord


    def get_structure(self, model=None, altloc=[], extra_fields=[]):
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the PDB file.
        
        This function parses standard base-10 PDB files as well as
        hybrid-36 PDB.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number.
            If this parameter is omitted, an :class:`AtomArrayStack`
            containing all models will be returned, even if the
            structure contains only one model.
        altloc : list of tuple, optional
            In case the structure contains *altloc* entries, those can be
            specified here:
            Each tuple consists of the following elements:

                - A chain ID, specifying the residue
                - A residue ID, specifying the residue
                - The desired *altoc* ID for the specified residue

            For each of the given residues the atoms with the given *altloc*
            ID are filtered.
            By default the location with the *altloc* ID "A" is used.
        extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the output array or stack.
            These are valid values:
            ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
            ``'charge'``.
        
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
            if model < 1:
                raise ValueError(
                    f"Requested model number {model} is smaller than 1"
                )
            elif model < last_model:
                line_filter = ( ( atom_line_i >= model_start_i[model-1] ) &
                                ( atom_line_i <  model_start_i[model  ] ) )
            elif model == last_model:
                line_filter = (atom_line_i >= model_start_i[model-1])
            else:
                raise ValueError(
                    f"Requested model number {model} is larger than the "
                    f"amount of models ({last_model})"
                )
            annot_i = coord_i = atom_line_i[line_filter]
            array = AtomArray(len(coord_i))
        
        # Create altloc array for the final filtering
        altloc_array = np.zeros(array.array_length(), dtype="U1")
        
        # Add optional annotation arrays
        for field in extra_fields:
            if field in ["atom_id", "charge"]:
                array.add_annotation(field, dtype=int)
            elif field in ["occupancy", "b_factor"]:
                array.add_annotation(field, dtype=float)
            else:
                raise ValueError(f"Unknown extra field: {field}")
        
        # Fill in annotation
        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            altloc_array[i] = line[16]
            array.chain_id[i] = line[21].upper().strip()
            array.res_id[i] = decode_hybrid36(line[22:26])
            array.ins_code[i] = line[26].strip()
            array.res_name[i] = line[17:20].strip()
            array.hetero[i] = (False if line[0:4] == "ATOM" else True)
            array.atom_name[i] = line[12:16].strip()
            array.element[i] = line[76:78].strip()

        # Replace empty strings for elements with guessed types
        # This is used e.g. for PDB files created by Gromacs
        def guess_element(atom_name):
            if atom_name.startswith(("H", "1H", "2H", "3H")):
                return 'H'
            return atom_name[0]

        if "" in array.element:
            rep_num = 0
            for idx in range(len(array.element)):
                if not array.element[idx]:
                    atom_name = array.atom_name[idx]
                    array.element[idx] = guess_element(atom_name)
                    rep_num += 1
            warn("{} elements were guessed from atom_name.".format(rep_num))
                            
        if extra_fields:
            for i, line_i in enumerate(annot_i):
                line = self.lines[line_i]
                if "atom_id" in extra_fields:
                    array.atom_id[i] = decode_hybrid36(line[6:11])
                if "occupancy" in extra_fields:
                    array.occupancy[i] = float(line[54:60].strip())
                if "b_factor" in extra_fields:
                    array.b_factor[i] = float(line[60:66].strip())
                if "charge" in extra_fields:
                    sign = -1 if line[79] == "-" else 1
                    array.charge[i] = (0 if line[78] == " "
                                       else int(line[78]) * sign)
        
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

        # Fill in box vectors
        # PDB does not support changing box dimensions. CRYST1 is a one-time
        # record so we can extract it directly
        for line in self.lines:
            if line.startswith("CRYST1"):
                len_a = float(line[6:15])
                len_b = float(line[15:24])
                len_c = float(line[24:33])
                alpha = np.deg2rad(float(line[33:40]))
                beta = np.deg2rad(float(line[40:47]))
                gamma = np.deg2rad(float(line[47:54]))
                box = vectors_from_unitcell(
                    len_a, len_b, len_c, alpha, beta, gamma
                )

                if isinstance(array, AtomArray):
                    array.box = box
                else:
                    array.box = np.repeat(
                        box[np.newaxis, ...], array.stack_depth(), axis=0
                    )
                break

        # Apply final filter and return
        return array[..., filter_altloc(array, altloc_array, altloc)]




    def set_structure(self, array, hybrid36=False):
        """
        Set the :class:`AtomArray` or :class:`AtomArrayStack` for the
        file.
        
        This makes also use of the optional annotation arrays
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
        If the atom array (stack) contains the annotation ``'atom_id'``,
        these values will be used for atom numbering instead of
        continuous numbering.
        
        Parameters
        ----------
        array : AtomArray or AtomArrayStack
            The array or stack to be saved into this file. If a stack
            is given, each array in the stack is saved as separate
            model.
        hybrid36: bool, optional
            Defines wether the file should be written in hybrid-36
            format.
        """
        annot_categories = array.get_annotation_categories()
        hetero = ["ATOM" if e == False else "HETATM" for e in array.hetero]
        # Check for optional annotation categories
        if "atom_id" in annot_categories:
            atom_id = array.atom_id
        else:
            atom_id = np.arange(1, array.array_length()+1)
        if "b_factor" in annot_categories:
            b_factor = array.b_factor
        else:
            b_factor = np.zeros(array.array_length())
        if "occupancy" in annot_categories:
            occupancy = array.occupancy
        else:
            occupancy = np.ones(array.array_length())
        if "charge" in annot_categories:
            charge = [ "+"+str(np.abs(charge)) if charge > 0 else
                      ("-"+str(np.abs(charge)) if charge < 0 else
                       "")
                      for charge in array.get_annotation("charge")]
        else:
            charge = [""] * array.array_length()

        # Do checks on atom array (stack)
        if hybrid36:
            max_atoms, max_residues \
                = max_hybrid36_number(5), max_hybrid36_number(4)
        else:
            max_atoms, max_residues = 99999, 9999
        if array.array_length() > max_atoms:
            warn(f"More then {max_atoms:,} atoms per model")
        if (array.res_id > max_residues).any():
            warn(f"Residue IDs exceed {max_residues:,}")
        if np.isnan(array.coord).any():
            raise ValueError("Coordinates contain 'NaN' values")

        if hybrid36:
            pdb_atom_id = [encode_hybrid36(i, 5).rjust(5) for i in atom_id]
            pdb_res_id = [encode_hybrid36(i, 4).rjust(4) for i in array.res_id]
        else:
            # Atom IDs are supported up to 99999,
            # but negative IDs are also possible
            pdb_atom_id = np.where(
                atom_id > 0,
                ((atom_id - 1) % 99999) + 1,
                atom_id
            )
            pdb_atom_id = ["{:>5d}".format(i) for i in pdb_atom_id]
            # Residue IDs are supported up to 9999,
            # but negative IDs are also possible
            pdb_res_id = np.where(
                array.res_id > 0,
                ((array.res_id - 1) % 9999) + 1,
                array.res_id
            )
            pdb_res_id = ["{:>4d}".format(i) for i in pdb_res_id]

        if isinstance(array, AtomArray):
            self.lines = [None] * array.array_length()
            for i in range(array.array_length()):
                self.lines[i] = ("{:6}".format(hetero[i]) + 
                                  pdb_atom_id[i] +
                                  " " +
                                  "{:4}".format(array.atom_name[i]) +
                                  " " +
                                  "{:3}".format(array.res_name[i]) +
                                  " " +
                                  "{:1}".format(array.chain_id[i]) +
                                  pdb_res_id[i] +
                                  "{:1}".format(array.ins_code[i]) +
                                  (" " * 3) +
                                  "{:>8.3f}".format(array.coord[i,0]) +
                                  "{:>8.3f}".format(array.coord[i,1]) +
                                  "{:>8.3f}".format(array.coord[i,2]) +
                                  "{:>6.2f}".format(occupancy[i]) +
                                  "{:>6.3f}".format(b_factor[i]) +
                                  (" " * 10) + 
                                  "{:2}".format(array.element[i]) +
                                  "{:2}".format(charge[i])
                                 )
        
        elif isinstance(array, AtomArrayStack):
            self.lines = []

            # The entire information, but the coordinates,
            # is equal for each model
            # Therefore template lines are created
            # which are afterwards applied for each model
            templines = [None] * array.array_length()
            for i in range(array.array_length()):
                templines[i] = ("{:6}".format(hetero[i]) + 
                                 pdb_atom_id[i] +
                                 " " +
                                 "{:4}".format(array.atom_name[i]) +
                                 " " +
                                 "{:3}".format(array.res_name[i]) +
                                 " " +
                                 "{:1}".format(array.chain_id[i]) +
                                 pdb_res_id[i] +
                                 "{:1}".format(array.ins_code[i]) +
                                 (" " * 27) +
                                 "{:>6.2f}".format(occupancy[i]) +
                                 "{:>6.3f}".format(b_factor[i]) +
                                 (" " * 10) +
                                 "{:2}".format(array.element[i]) +
                                 "{:2}".format(charge[i])
                                )
            for i in range(array.stack_depth()):
                #Fill in coordinates for each model
                self.lines.append("{:5}{:>9d}".format("MODEL", i+1))
                modellines = copy.copy(templines)
                for j, line in enumerate(modellines):
                    # Insert coordinates
                    line = (line[:30]
                            + "{:>8.3f}{:>8.3f}{:>8.3f}".format(
                                    array.coord[i,j,0],
                                    array.coord[i,j,1],
                                    array.coord[i,j,2])
                            + line[54:] )
                    modellines[j] = line
                self.lines.extend(modellines)
                self.lines.append("ENDMDL")

        # prepend a single CRYST1 record if we have box information
        if array.box is not None:
            box = array.box
            if len(box.shape) == 3:
                box = box[0]
            unitcell = unitcell_from_vectors(box)
            self.lines.insert(0, "CRYST1" +
                              "{:>9.3f}".format(unitcell[0]) +
                              "{:>9.3f}".format(unitcell[1]) +
                              "{:>9.3f}".format(unitcell[2]) +
                              "{:>7.2f}".format(np.rad2deg(unitcell[3])) +
                              "{:>7.2f}".format(np.rad2deg(unitcell[4])) +
                              "{:>7.2f}".format(np.rad2deg(unitcell[5])) +
                              " P 1           1")

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