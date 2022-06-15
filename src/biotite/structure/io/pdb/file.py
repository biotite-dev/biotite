# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann, Daniel Bauer, Claude J. Rogers"
__all__ = ["PDBFile"]

import warnings
import numpy as np
from ...atoms import AtomArray, AtomArrayStack
from ...bonds import BondList, connect_via_residue_names
from ...box import vectors_from_unitcell, unitcell_from_vectors
from ....file import TextFile, InvalidFileError
from ..general import _guess_element as guess_element
from ...error import BadStructureError
from ...filter import (
    filter_first_altloc,
    filter_highest_occupancy_altloc,
    filter_solvent,
)
from .hybrid36 import encode_hybrid36, decode_hybrid36, max_hybrid36_number
import copy
from warnings import warn

# slice objects for readability
# ATOM/HETATM
_record = slice(0, 6)
_atom_id = slice(6, 11)
_atom_name = slice(12, 16)
_alt_loc = slice(16, 17)
_res_name = slice(17, 20)
_chain_id = slice(21, 22)
_res_id = slice(22, 26)
_ins_code = slice(26, 27)
_coord_x = slice(30, 38)
_coord_y = slice(38, 46)
_coord_z = slice(46, 54)
_occupancy = slice(54, 60)
_temp_f = slice(60, 66)
_element = slice(76, 78)
_charge = slice(78, 80)
# CRYST1
_a = slice(6, 15)
_b = slice(15, 24)
_c = slice(24, 33)
_alpha = slice(33, 40)
_beta = slice(40, 47)
_gamma = slice(47, 54)


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
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1l2y.pdb"))
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = PDBFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.pdb"))
    """
    @classmethod
    def read(cls, file):
        file = super().read(file)
        # Pad lines with whitespace if lines are shorter
        # than the required 80 characters
        file.lines = [line.ljust(80) for line in file.lines]
        return file


    def get_model_count(self):
        """
        Get the number of models contained in the PDB file.

        Returns
        -------
        model_count : int
            The number of models.
        """
        model_count = 0
        for line in self.lines:
            if line.startswith("MODEL"):
                model_count += 1
        
        if model_count == 0:
            # It could be an empty file or a file with a single model,
            # where the 'MODEL' line is missing
            for line in self.lines:
                if line.startswith(("ATOM", "HETATM")):
                    return 1
            return 0
        else:
            return model_count
    
    def get_coord(self, model=None):
        """
        Get only the coordinates of the PDB file.
        
        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return a
            2D coordinate array from the atoms corresponding to the
            given model number (starting at 1).
            Negative values are used to index models starting from the
            last model insted of the first model.
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
        *altloc* IDs, while `get_coord()` does not.
        
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
        >>> from tempfile import gettempdir
        >>> file_names = []
        >>> for i in range(atom_array_stack.stack_depth()):
        ...     pdb_file = PDBFile()
        ...     pdb_file.set_structure(atom_array_stack[i])
        ...     file_name = os.path.join(gettempdir(), f"model_{i+1}.pdb")
        ...     pdb_file.write(file_name)
        ...     file_names.append(file_name)
        >>> print(file_names)
        ['...model_1.pdb', '...model_2.pdb', ..., '...model_38.pdb']

        Now the PDB files are used to create an :class:`AtomArrayStack`,
        where each model represents a different model.

        Construct a new :class:`AtomArrayStack` with annotations taken
        from one of the created files used as template and coordinates
        from all of the PDB files.

        >>> template_file = PDBFile.read(file_names[0])
        >>> template = template_file.get_structure()
        >>> coord = []
        >>> for i, file_name in enumerate(file_names):
        ...     pdb_file = PDBFile.read(file_name)
        ...     coord.append(pdb_file.get_coord(model=1))
        >>> new_stack = from_template(template, np.array(coord))

        The newly created :class:`AtomArrayStack` should now be equal to
        the :class:`AtomArrayStack` the PDB files were created from.

        >>> print(np.allclose(new_stack.coord, atom_array_stack.coord))
        True
        """
        # Line indices where a new model starts
        model_start_i = np.array([i for i in range(len(self.lines))
                                  if self.lines[i].startswith("MODEL")],
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

    def get_structure(self, model=None, altloc="first", extra_fields=[],
                      include_bonds=False):
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the PDB file.
        
        This function parses standard base-10 PDB files as well as
        hybrid-36 PDB.
        
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
        altloc : {'first', 'occupancy', 'all'}
            This parameter defines how *altloc* IDs are handled:
                - ``'first'`` - Use atoms that have the first
                  *altloc* ID appearing in a residue.
                - ``'occupancy'`` - Use atoms that have the *altloc* ID
                  with the highest occupancy for a residue.
                - ``'all'`` - Use all atoms.
                  Note that this leads to duplicate atoms.
                  When this option is chosen, the ``altloc_id``
                  annotation array is added to the returned structure.
        extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the output array or stack.
            These are valid values:
            ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and
            ``'charge'``.
        include_bonds : bool, optional
            If set to true, a :class:`BondList` will be created for the
            resulting :class:`AtomArray` containing the bond information
            from the file.
            All bonds have :attr:`BondType.ANY`, since the PDB format
            does not support bond orders.
        
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
        
        # Create mandatory and optional annotation arrays
        chain_id  = np.zeros(array.array_length(), array.chain_id.dtype)
        res_id    = np.zeros(array.array_length(), array.res_id.dtype)
        ins_code  = np.zeros(array.array_length(), array.ins_code.dtype)
        res_name  = np.zeros(array.array_length(), array.res_name.dtype)
        hetero    = np.zeros(array.array_length(), array.hetero.dtype)
        atom_name = np.zeros(array.array_length(), array.atom_name.dtype)
        element   = np.zeros(array.array_length(), array.element.dtype)
        atom_id_raw = np.zeros(array.array_length(), "U5")
        charge_raw  = np.zeros(array.array_length(), "U2")
        occupancy = np.zeros(array.array_length(), float)
        b_factor  = np.zeros(array.array_length(), float)
        altloc_id = np.zeros(array.array_length(), dtype="U1")

        # Fill annotation array
        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            chain_id[i] = line[_chain_id].upper().strip()
            res_id[i] = decode_hybrid36(line[_res_id])
            ins_code[i] = line[_ins_code].strip()
            res_name[i] = line[_res_name].strip()
            hetero[i] = line[_record] == "HETATM"
            atom_name[i] = line[_atom_name].strip()
            element[i] = line[_element].strip()
            altloc_id[i] = line[_alt_loc]
            atom_id_raw[i] = line[_atom_id]
            charge_raw[i] = line[_charge][::-1]  # turn "1-" into "-1"
            occupancy[i] = float(line[_occupancy].strip())
            b_factor[i] = float(line[_temp_f].strip())
        
        if include_bonds or \
            (extra_fields is not None and "atom_id" in extra_fields):
                # The atom IDs are only required in these two cases
                atom_id = np.array(
                    [decode_hybrid36(raw_id.item()) for raw_id in atom_id_raw],
                    dtype=int
                )
        else:
            atom_id = None
        
        # Add annotation arrays to atom array (stack)
        array.chain_id = chain_id
        array.res_id = res_id
        array.ins_code = ins_code
        array.res_name = res_name
        array.hetero = hetero
        array.atom_name = atom_name
        array.element = element

        for field in (extra_fields if extra_fields is not None else []):
            if field == "atom_id":
                # Copy is necessary to avoid double masking in 
                # later altloc ID filtering
                array.set_annotation("atom_id", atom_id.copy())
            elif field == "charge":
                charge = np.array(charge_raw)
                array.set_annotation("charge", np.where(
                    charge == "  ", "0", charge
                ).astype(int))
            elif field == "occupancy":
                array.set_annotation("occupancy", occupancy)
            elif field == "b_factor":
                array.set_annotation("b_factor", b_factor)
            else:
                raise ValueError(f"Unknown extra field: {field}")

        # Replace empty strings for elements with guessed types
        # This is used e.g. for PDB files created by Gromacs
        if "" in array.element:
            rep_num = 0
            for idx in range(len(array.element)):
                if not array.element[idx]:
                    atom_name = array.atom_name[idx]
                    array.element[idx] = guess_element(atom_name)
                    rep_num += 1
            warn("{} elements were guessed from atom_name.".format(rep_num))
        
        # Fill in coordinates
        if isinstance(array, AtomArray):
            for i, line_i in enumerate(coord_i):
                line = self.lines[line_i]
                array.coord[i, 0] = float(line[_coord_x])
                array.coord[i, 1] = float(line[_coord_y])
                array.coord[i, 2] = float(line[_coord_z])
                
        elif isinstance(array, AtomArrayStack):
            m = 0
            i = 0
            for line_i in atom_line_i:
                if m < len(model_start_i)-1 and line_i > model_start_i[m+1]:
                    m += 1
                    i = 0
                line = self.lines[line_i]
                array.coord[m, i, 0] = float(line[_coord_x])
                array.coord[m, i, 1] = float(line[_coord_y])
                array.coord[m, i, 2] = float(line[_coord_z])
                i += 1

        # Fill in box vectors
        # PDB does not support changing box dimensions. CRYST1 is a one-time
        # record so we can extract it directly
        for line in self.lines:
            if line.startswith("CRYST1"):
                try:
                    len_a = float(line[_a])
                    len_b = float(line[_b])
                    len_c = float(line[_c])
                    alpha = np.deg2rad(float(line[_alpha]))
                    beta = np.deg2rad(float(line[_beta]))
                    gamma = np.deg2rad(float(line[_gamma]))
                    box = vectors_from_unitcell(
                        len_a, len_b, len_c, alpha, beta, gamma
                    )
                except ValueError:
                    # File contains invalid 'CRYST1' record
                    warnings.warn(
                        "File contains invalid 'CRYST1' record, box is ignored"
                    )
                    box = None

                if isinstance(array, AtomArray):
                    array.box = box
                else:
                    array.box = np.repeat(
                        box[np.newaxis, ...], array.stack_depth(), axis=0
                    )
                break  

        # Filter altloc IDs
        if altloc == "occupancy":
            filter = filter_highest_occupancy_altloc(
                array, altloc_id, occupancy
            )
            array = array[..., filter]
            atom_id = atom_id[filter] if atom_id is not None else None
        elif altloc == "first":
            filter = filter_first_altloc(array, altloc_id)
            array = array[..., filter]
            atom_id = atom_id[filter] if atom_id is not None else None
        elif altloc == "all":
            array.set_annotation("altloc_id", altloc_id)
        else:
            raise ValueError(f"'{altloc}' is not a valid 'altloc' option")
        
        # Read bonds
        if include_bonds:
            bond_list = self._get_bonds(atom_id)
            bond_list = bond_list.merge(connect_via_residue_names(
                array,
                # The information for non-hetero residues and water
                # are not part of CONECT records
                (~array.hetero) | filter_solvent(array)
            ))
            # Remove bond order from inter residue bonds for consistency
            bond_list.remove_bond_order()
            array.bonds = bond_list  
        
        return array

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
        
        Notes
        -----
        If `array` has an associated :class:`BondList`, ``CONECT``
        records are also written for all non-water hetero residues
        and all inter-residue connections.
        """
        natoms = array.array_length()
        annot_categories = array.get_annotation_categories()
        record = np.char.array(np.where(array.hetero, "HETATM", "ATOM"))
        # Check for optional annotation categories
        if "atom_id" in annot_categories:
            atom_id = array.atom_id
        else:
            atom_id = np.arange(1, natoms + 1)
        if "b_factor" in annot_categories:
            b_factor = np.char.array([f"{b:>6.2f}" for b in array.b_factor])
        else:
            b_factor = np.char.array(np.full(natoms, "  0.00", dtype="U6"))
        if "occupancy" in annot_categories:
            occupancy = np.char.array([f"{o:>6.2f}" for o in array.occupancy])
        else:
            occupancy = np.char.array(np.full(natoms, "  1.00", dtype="U6"))
        if "charge" in annot_categories:
            charge = np.char.array(
                [str(np.abs(charge)) + "+" if charge > 0 else
                 (str(np.abs(charge)) + "-" if charge < 0 else "")
                 for charge in array.get_annotation("charge")]
            )
        else:
            charge = np.char.array(np.full(natoms, "  ", dtype="U2"))

        # Do checks on atom array (stack)
        if hybrid36:
            max_atoms = max_hybrid36_number(5)
            max_residues = max_hybrid36_number(4)
        else:
            max_atoms, max_residues = 99999, 9999
        if array.array_length() > max_atoms:
            warn(f"More then {max_atoms:,} atoms per model")
        if (array.res_id > max_residues).any():
            warn(f"Residue IDs exceed {max_residues:,}")
        if np.isnan(array.coord).any():
            raise ValueError("Coordinates contain 'NaN' values")
        if any([len(name) > 1 for name in array.chain_id]):
            raise ValueError("Some chain IDs exceed 1 character")
        if any([len(name) > 3 for name in array.res_name]):
            raise ValueError("Some residue names exceed 3 characters")
        if any([len(name) > 4 for name in array.atom_name]):
            raise ValueError("Some atom names exceed 4 characters")

        if hybrid36:
            pdb_atom_id = np.char.array(
                [encode_hybrid36(i, 5) for i in atom_id]
            )
            pdb_res_id = np.char.array(
                [encode_hybrid36(i, 4) for i in array.res_id]
            )
        else:
            # Atom IDs are supported up to 99999,
            # but negative IDs are also possible
            pdb_atom_id = np.char.array(np.where(
                atom_id > 0,
                ((atom_id - 1) % 99999) + 1,
                atom_id
            ).astype(str))
            # Residue IDs are supported up to 9999,
            # but negative IDs are also possible
            pdb_res_id = np.char.array(np.where(
                array.res_id > 0,
                ((array.res_id - 1) % 9999) + 1,
                array.res_id
            ).astype(str))
        
        names = np.char.array(
            [f" {atm}" if len(elem) == 1 and len(atm) < 4 else atm
             for atm, elem in zip(array.atom_name, array.element)]
        )
        res_names = np.char.array(array.res_name)
        chain_ids = np.char.array(array.chain_id)
        ins_codes = np.char.array(array.ins_code)
        spaces = np.char.array(np.full(natoms, " ", dtype="U1"))
        elements = np.char.array(array.element)

        first_half = (
            record.ljust(6) +
            pdb_atom_id.rjust(5) +
            spaces +
            names.ljust(4) +
            spaces + res_names.rjust(3) + spaces + chain_ids +
            pdb_res_id.rjust(4) + ins_codes.rjust(1)
        )

        second_half = (
            occupancy + b_factor + 10 * spaces +
            elements.rjust(2) + charge.rjust(2)
        )

        coords = array.coord
        if coords.ndim == 2:
            coords = coords[np.newaxis, ...]
        
        self.lines = []
        # Prepend a single CRYST1 record if we have box information
        if array.box is not None:
            box = array.box
            if len(box.shape) == 3:
                box = box[0]
            a, b, c, alpha, beta, gamma = unitcell_from_vectors(box)
            self.lines.append(
                f"CRYST1{a:>9.3f}{b:>9.3f}{c:>9.3f}"
                f"{np.rad2deg(alpha):>7.2f}{np.rad2deg(beta):>7.2f}"
                f"{np.rad2deg(gamma):>7.2f} P 1           1"
            )
        is_stack = coords.shape[0] > 1
        for model_num, coord_i in enumerate(coords, start=1):
            # for an ArrayStack, this is run once
            # only add model lines if is_stack
            if is_stack:
                self.lines.append(f"MODEL     {model_num:4}")
            # Bundle non-coordinate data to simplify iteration
            self.lines.extend(
                [f"{start:27}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{end:26}"
                 for start, (x, y, z), end in
                 zip(first_half, coord_i, second_half)]
            )
            if is_stack:
                self.lines.append("ENDMDL")
        
        # Add CONECT records if bonds are present
        if array.bonds is not None:
            # Only non-water hetero records and connections between
            # residues are added to the records
            hetero_indices = np.where(array.hetero & ~filter_solvent(array))[0]
            bond_array = array.bonds.as_array()
            bond_array = bond_array[
                np.isin(bond_array[:,0], hetero_indices) |
                np.isin(bond_array[:,1], hetero_indices) |
                (array.res_id  [bond_array[:,0]] != array.res_id  [bond_array[:,1]]) |
                (array.chain_id[bond_array[:,0]] != array.chain_id[bond_array[:,1]])
            ]
            self._set_bonds(
                BondList(array.array_length(), bond_array), atom_id
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

    def _get_bonds(self, atom_ids):
        conect_lines = [line for line in self.lines
                        if line.startswith("CONECT")]
        
        # Mapping from atom ids to indices in an AtomArray
        atom_id_to_index = np.zeros(atom_ids[-1]+1, dtype=int)
        try:
            for i, id in enumerate(atom_ids):
                atom_id_to_index[id] = i
        except IndexError as e:
            raise InvalidFileError(
                "Atom IDs are not strictly increasing"
            ) from e

        bonds = []
        for line in conect_lines:
            center_id = atom_id_to_index[int(line[6 : 11])]
            for i in range(11, 31, 5):
                id_string = line[i : i+5]
                try:
                    id = atom_id_to_index[int(id_string)]
                except ValueError:
                    # String is empty -> no further IDs
                    break
                bonds.append((center_id, id))
        
        # The length of the 'atom_ids' array
        # is equal to the length of the AtomArray
        return BondList(len(atom_ids), np.array(bonds, dtype=np.uint32))

    def _set_bonds(self, bond_list, atom_ids):
        # Bond type is unused since PDB does not support bond orders
        bonds, _ = bond_list.get_all_bonds()

        for center_i, bonded_indices in enumerate(bonds):
            n_added = 0
            for bonded_i in bonded_indices:
                if bonded_i == -1:
                    # Reached padding values
                    break
                if n_added == 0:
                    # Add new record
                    line = f"CONECT{atom_ids[center_i]:>5d}"
                line += f"{atom_ids[bonded_i]:>5d}"
                n_added += 1
                if n_added == 4:
                    # Only a maximum of 4 bond partners can be put
                    # into a single line
                    # If there are more, use an extra record
                    n_added = 0
                    self.lines.append(line)
            if n_added > 0:
                self.lines.append(line)