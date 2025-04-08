# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann, Daniel Bauer, Claude J. Rogers"
__all__ = ["PDBFile"]

import warnings
from collections import namedtuple
import numpy as np
from biotite.file import InvalidFileError, TextFile
from biotite.structure.atoms import AtomArray, AtomArrayStack, repeat
from biotite.structure.bonds import BondList, connect_via_residue_names
from biotite.structure.box import unitcell_from_vectors, vectors_from_unitcell
from biotite.structure.error import BadStructureError
from biotite.structure.filter import (
    filter_first_altloc,
    filter_highest_occupancy_altloc,
    filter_solvent,
)
from biotite.structure.io.pdb.hybrid36 import (
    decode_hybrid36,
    encode_hybrid36,
    max_hybrid36_number,
)
from biotite.structure.io.util import number_of_integer_digits
from biotite.structure.repair import infer_elements
from biotite.structure.util import matrix_rotate

_PDB_MAX_ATOMS = 99999
_PDB_MAX_RESIDUES = 9999

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
_space = slice(55, 66)
_z = slice(66, 70)


class PDBFile(TextFile):
    r"""
    This class represents a PDB file.

    The usage of :mod:`biotite.structure.io.pdbx` is encouraged in favor
    of this class.

    This class only provides support for reading/writing the pure atom
    information (*ATOM*, *HETATM*, *MODEL* and *ENDMDL* records). *TER*
    records cannot be written.
    Additionally, *REMARK* records can be read

    See Also
    --------
    CIFFile : Interface to CIF files, a modern replacement for PDB files.
    BinaryCIFFile : Interface to BinaryCIF files, a binary variant of CIF files.

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
        file._index_models_and_atoms()
        return file

    def get_remark(self, number):
        r"""
        Get the lines containing the *REMARK* records with the given
        `number`.

        Parameters
        ----------
        number : int
            The *REMARK* number, i.e. the `XXX` in ``REMARK XXX``.

        Returns
        -------
        remark_lines : None or list of str
            The content of the selected *REMARK* lines.
            Each line is an element of this list.
            The ``REMARK XXX `` part of each line is omitted.
            Furthermore, the first line, which always must be empty, is
            not included.
            ``None`` is returned, if the selected *REMARK* records do not
            exist in the file.

        Examples
        --------

        >>> import os.path
        >>> file = PDBFile.read(os.path.join(path_to_structures, "1l2y.pdb"))
        >>> remarks = file.get_remark(900)
        >>> print("\n".join(remarks))
        RELATED ENTRIES
        RELATED ID: 5292   RELATED DB: BMRB
        BMRB 5292 IS CHEMICAL SHIFTS FOR TC5B IN BUFFER AND BUFFER
        CONTAINING 30 VOL-% TFE.
        RELATED ID: 1JRJ   RELATED DB: PDB
        1JRJ IS AN ANALAGOUS C-TERMINAL STRUCTURE.
        >>> nonexistent_remark = file.get_remark(999)
        >>> print(nonexistent_remark)
        None
        """
        CONTENT_START_COLUMN = 11

        # in case a non-integer is accidentally given
        number = int(number)
        if number < 0 or number > 999:
            raise ValueError("The number must be in range 0-999")

        remark_string = f"REMARK {number:>3d}"
        # Find lines and omit ``REMARK XXX `` part
        remark_lines = [
            line[CONTENT_START_COLUMN:]
            for line in self.lines
            if line.startswith(remark_string)
        ]
        if len(remark_lines) == 0:
            return None
        # Remove first empty line
        remark_lines = remark_lines[1:]
        return remark_lines

    def get_model_count(self):
        """
        Get the number of models contained in the PDB file.

        Returns
        -------
        model_count : int
            The number of models.
        """
        return len(self._model_start_i)

    def get_coord(self, model=None):
        """
        Get only the coordinates from the PDB file.

        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return a
            2D coordinate array from the atoms corresponding to the
            given model number (starting at 1).
            Negative values are used to index models starting from the
            last model instead of the first model.
            If this parameter is omitted, an 3D coordinate array
            containing all models will be returned, even if
            the structure contains only one model.

        Returns
        -------
        coord : ndarray, shape=(m,n,3) or shape=(n,3), dtype=float
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
        if model is None:
            coord = np.zeros(
                (len(self._model_start_i), self._get_model_length(), 3),
                dtype=np.float32,
            )
            m = 0
            i = 0
            for line_i in self._atom_line_i:
                if (
                    m < len(self._model_start_i) - 1
                    and line_i > self._model_start_i[m + 1]
                ):
                    m += 1
                    i = 0
                line = self.lines[line_i]
                coord[m, i, 0] = float(line[_coord_x])
                coord[m, i, 1] = float(line[_coord_y])
                coord[m, i, 2] = float(line[_coord_z])
                i += 1
            return coord

        else:
            coord_i = self._get_atom_record_indices_for_model(model)
            coord = np.zeros((len(coord_i), 3), dtype=np.float32)
            for i, line_i in enumerate(coord_i):
                line = self.lines[line_i]
                coord[i, 0] = float(line[_coord_x])
                coord[i, 1] = float(line[_coord_y])
                coord[i, 2] = float(line[_coord_z])
            return coord

    def get_b_factor(self, model=None):
        """
        Get only the B-factors from the PDB file.

        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return a
            1D B-factor array from the atoms corresponding to the
            given model number (starting at 1).
            Negative values are used to index models starting from the
            last model instead of the first model.
            If this parameter is omitted, an 2D B-factor array
            containing all models will be returned, even if
            the structure contains only one model.

        Returns
        -------
        b_factor : ndarray, shape=(m,n) or shape=(n,), dtype=float
            The B-factors read from the ATOM and HETATM records of the
            file.

        Notes
        -----
        Note that :func:`get_b_factor()` may output more B-factors
        than the atom array (stack) from the corresponding
        :func:`get_structure()` call has atoms.
        The reason for this is, that :func:`get_structure()` filters
        *altloc* IDs, while `get_b_factor()` does not.
        """
        if model is None:
            b_factor = np.zeros(
                (len(self._model_start_i), self._get_model_length()), dtype=np.float32
            )
            m = 0
            i = 0
            for line_i in self._atom_line_i:
                if (
                    m < len(self._model_start_i) - 1
                    and line_i > self._model_start_i[m + 1]
                ):
                    m += 1
                    i = 0
                line = self.lines[line_i]
                b_factor[m, i] = float(line[_temp_f])
                i += 1
            return b_factor

        else:
            b_factor_i = self._get_atom_record_indices_for_model(model)
            b_factor = np.zeros(len(b_factor_i), dtype=np.float32)
            for i, line_i in enumerate(b_factor_i):
                line = self.lines[line_i]
                b_factor[i] = float(line[_temp_f])
            return b_factor

    def get_structure(
        self, model=None, altloc="first", extra_fields=[], include_bonds=False
    ):
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
            last model instead of the first model.
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
            Bonds, whose order could not be determined from the
            *Chemical Component Dictionary*
            (e.g. especially inter-residue bonds),
            have :attr:`BondType.ANY`, since the PDB format itself does
            not support bond orders.

        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """
        if model is None:
            depth = len(self._model_start_i)
            length = self._get_model_length()
            array = AtomArrayStack(depth, length)
            # Record indices for annotation determination
            # Annotation is determined from model 1
            annot_i = self._get_atom_record_indices_for_model(1)
            # Record indices for coordinate determination
            coord_i = self._atom_line_i

        else:
            annot_i = coord_i = self._get_atom_record_indices_for_model(model)
            array = AtomArray(len(coord_i))

        # Create mandatory and optional annotation arrays
        chain_id = np.zeros(array.array_length(), array.chain_id.dtype)
        res_id = np.zeros(array.array_length(), array.res_id.dtype)
        ins_code = np.zeros(array.array_length(), array.ins_code.dtype)
        res_name = np.zeros(array.array_length(), array.res_name.dtype)
        hetero = np.zeros(array.array_length(), array.hetero.dtype)
        atom_name = np.zeros(array.array_length(), array.atom_name.dtype)
        element = np.zeros(array.array_length(), array.element.dtype)
        atom_id_raw = np.zeros(array.array_length(), "U5")
        charge_raw = np.zeros(array.array_length(), "U2")
        occupancy = np.zeros(array.array_length(), float)
        b_factor = np.zeros(array.array_length(), float)
        altloc_id = np.zeros(array.array_length(), dtype="U1")

        # Fill annotation array
        # i is index in array, line_i is line index
        for i, line_i in enumerate(annot_i):
            line = self.lines[line_i]
            chain_id[i] = line[_chain_id].strip()
            res_id[i] = decode_hybrid36(line[_res_id])
            ins_code[i] = line[_ins_code].strip()
            res_name[i] = line[_res_name].strip()
            hetero[i] = line[_record] == "HETATM"
            atom_name[i] = line[_atom_name].strip()
            element[i] = line[_element].strip()
            altloc_id[i] = line[_alt_loc]
            atom_id_raw[i] = line[_atom_id]
            # turn "1-" into "-1", if necessary
            if line[_charge][0] in "+-":
                charge_raw[i] = line[_charge]
            else:
                charge_raw[i] = line[_charge][::-1]
            occupancy[i] = float(line[_occupancy].strip())
            b_factor[i] = float(line[_temp_f].strip())

        if include_bonds or (extra_fields is not None and "atom_id" in extra_fields):
            # The atom IDs are only required in these two cases
            atom_id = np.array(
                [decode_hybrid36(raw_id.item()) for raw_id in atom_id_raw], dtype=int
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

        for field in extra_fields if extra_fields is not None else []:
            if field == "atom_id":
                # Copy is necessary to avoid double masking in
                # later altloc ID filtering
                array.set_annotation("atom_id", atom_id.copy())
            elif field == "charge":
                charge = np.array(charge_raw)
                array.set_annotation(
                    "charge", np.where(charge == "  ", "0", charge).astype(int)
                )
            elif field == "occupancy":
                array.set_annotation("occupancy", occupancy)
            elif field == "b_factor":
                array.set_annotation("b_factor", b_factor)
            else:
                raise ValueError(f"Unknown extra field: {field}")

        # Replace empty strings for elements with guessed types
        # This is used e.g. for PDB files created by Gromacs
        empty_element_mask = array.element == ""
        if empty_element_mask.any():
            warnings.warn(
                f"{np.count_nonzero(empty_element_mask)} elements "
                "were guessed from atom name"
            )
            array.element[empty_element_mask] = infer_elements(
                array.atom_name[empty_element_mask]
            )

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
            for line_i in self._atom_line_i:
                if (
                    m < len(self._model_start_i) - 1
                    and line_i > self._model_start_i[m + 1]
                ):
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
                    box = vectors_from_unitcell(len_a, len_b, len_c, alpha, beta, gamma)
                except ValueError:
                    # File contains invalid 'CRYST1' record
                    warnings.warn(
                        "File contains invalid 'CRYST1' record, box is ignored"
                    )
                    break

                if isinstance(array, AtomArray):
                    array.box = box
                else:
                    array.box = np.repeat(
                        box[np.newaxis, ...], array.stack_depth(), axis=0
                    )
                break

        # Filter altloc IDs
        if altloc == "occupancy":
            filter = filter_highest_occupancy_altloc(array, altloc_id, occupancy)
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
            bond_list = bond_list.merge(connect_via_residue_names(array))
            array.bonds = bond_list

        return array

    def get_space_group(self):
        """
        Extract the space group and Z value from the CRYST1 record.

        Returns
        -------
        space_group : str
            The extracted space group.
        z_val : int
            The extracted Z value.
        """
        # Initialize the namedtuple
        SpaceGroupInfo = namedtuple("SpaceGroupInfo", ["space_group", "z_val"])

        # CRYST1 is a one-time record so we can extract it directly
        for line in self.lines:
            if line.startswith("CRYST1"):
                try:
                    # Extract space group and Z value
                    space_group = str(line[_space])
                    z_val = int(line[_z])
                except ValueError:
                    # File contains invalid 'CRYST1' record
                    raise InvalidFileError(
                        "File does not contain valid space group and/or Z values"
                    )
                    # Set default values
                    space_group = "P 1"
                    z_val = 1
                break
        return SpaceGroupInfo(space_group=space_group, z_val=z_val)

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
        hybrid36 : bool, optional
            Defines wether the file should be written in hybrid-36
            format.

        Notes
        -----
        If `array` has an associated :class:`BondList`, ``CONECT``
        records are also written for all non-water hetero residues
        and all inter-residue connections.
        """
        _check_pdb_compatibility(array, hybrid36)

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
                [
                    str(np.abs(charge)) + "+"
                    if charge > 0
                    else (str(np.abs(charge)) + "-" if charge < 0 else "")
                    for charge in array.get_annotation("charge")
                ]
            )
        else:
            charge = np.char.array(np.full(natoms, "  ", dtype="U2"))

        if hybrid36:
            pdb_atom_id = np.char.array([encode_hybrid36(i, 5) for i in atom_id])
            pdb_res_id = np.char.array([encode_hybrid36(i, 4) for i in array.res_id])
        else:
            # Atom IDs are supported up to 99999,
            # but negative IDs are also possible
            pdb_atom_id = np.char.array(
                np.where(
                    atom_id > 0, ((atom_id - 1) % _PDB_MAX_ATOMS) + 1, atom_id
                ).astype(str)
            )
            # Residue IDs are supported up to 9999,
            # but negative IDs are also possible
            pdb_res_id = np.char.array(
                np.where(
                    array.res_id > 0,
                    ((array.res_id - 1) % _PDB_MAX_RESIDUES) + 1,
                    array.res_id,
                ).astype(str)
            )

        names = np.char.array(
            [
                f" {atm}" if len(elem) == 1 and len(atm) < 4 else atm
                for atm, elem in zip(array.atom_name, array.element)
            ]
        )
        res_names = np.char.array(array.res_name)
        chain_ids = np.char.array(array.chain_id)
        ins_codes = np.char.array(array.ins_code)
        spaces = np.char.array(np.full(natoms, " ", dtype="U1"))
        elements = np.char.array(array.element)

        first_half = (
            record.ljust(6)
            + pdb_atom_id.rjust(5)
            + spaces
            + names.ljust(4)
            + spaces
            + res_names.rjust(3)
            + spaces
            + chain_ids
            + pdb_res_id.rjust(4)
            + ins_codes.rjust(1)
        )

        second_half = (
            occupancy + b_factor + 10 * spaces + elements.rjust(2) + charge.rjust(2)
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
                f"{np.rad2deg(gamma):>7.2f} P 1           1          "
            )
        is_stack = coords.shape[0] > 1
        for model_num, coord_i in enumerate(coords, start=1):
            # for an ArrayStack, this is run once
            # only add model lines if is_stack
            if is_stack:
                self.lines.append(f"MODEL     {model_num:4}")
            # Bundle non-coordinate data to simplify iteration
            self.lines.extend(
                [
                    f"{start:27}   {x:>8.3f}{y:>8.3f}{z:>8.3f}{end:26}"
                    for start, (x, y, z), end in zip(first_half, coord_i, second_half)
                ]
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
                np.isin(bond_array[:, 0], hetero_indices)
                | np.isin(bond_array[:, 1], hetero_indices)
                | (array.res_id[bond_array[:, 0]] != array.res_id[bond_array[:, 1]])
                | (array.chain_id[bond_array[:, 0]] != array.chain_id[bond_array[:, 1]])
            ]
            self._set_bonds(BondList(array.array_length(), bond_array), pdb_atom_id)

        self._index_models_and_atoms()

    def set_space_group(self, info):
        """
        Update the CRYST1 record with the provided space group and Z value.

        Parameters
        ----------
        info : tuple(str, int) or SpaceGroupInfo
            Contains the space group and Z-value.
        """
        for i, line in enumerate(self.lines):
            if line.startswith("CRYST1"):
                try:
                    # Format the replacement string
                    space_group_str = info.space_group.ljust(11)
                    z_val_str = str(info.z_val).rjust(4)

                    # Replace the existing CRYST1 record
                    self.lines[i] = line[:55] + space_group_str + z_val_str + line[70:]
                except (ValueError, AttributeError) as e:
                    # Raise an exception with context
                    raise AttributeError(
                        f"Failed to update CRYST1 record. "
                        f"Line: {line.strip()} | Error: {e}"
                    )
                break

    def list_assemblies(self):
        """
        List the biological assemblies that are available for the
        structure in the given file.

        This function receives the data from the ``REMARK 300`` records
        in the file.
        Consequently, this remark must be present in the file.

        Returns
        -------
        assemblies : list of str
            A list that contains the available assembly IDs.

        Examples
        --------
        >>> import os.path
        >>> file = PDBFile.read(os.path.join(path_to_structures, "1f2n.pdb"))
        >>> print(file.list_assemblies())
        ['1']
        """
        # Get remarks listing available assemblies
        remark_lines = self.get_remark(300)
        if remark_lines is None:
            raise InvalidFileError(
                "File does not contain assembly information (REMARK 300)"
            )
        return [assembly_id.strip() for assembly_id in remark_lines[0][12:].split(",")]

    def get_assembly(
        self,
        assembly_id=None,
        model=None,
        altloc="first",
        extra_fields=[],
        include_bonds=False,
    ):
        """
        Build the given biological assembly.

        This function receives the data from ``REMARK 350`` records in
        the file.
        Consequently, this remark must be present in the file.

        Parameters
        ----------
        assembly_id : str
            The assembly to build.
            Available assembly IDs can be obtained via
            :func:`list_assemblies()`.
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model instead of the first model.
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
            Bonds, whose order could not be determined from the
            *Chemical Component Dictionary*
            (e.g. especially inter-residue bonds),
            have :attr:`BondType.ANY`, since the PDB format itself does
            not support bond orders.

        Returns
        -------
        assembly : AtomArray or AtomArrayStack
            The assembly.
            The return type depends on the `model` parameter.
            Contains the `sym_id` annotation, which enumerates the copies of the
            asymmetric unit in the assembly.

        Examples
        --------

        >>> import os.path
        >>> file = PDBFile.read(os.path.join(path_to_structures, "1f2n.pdb"))
        >>> assembly = file.get_assembly(model=1)
        """
        # Get base structure
        structure = self.get_structure(
            model,
            altloc,
            extra_fields,
            include_bonds,
        )

        # Get lines containing transformations for chosen assembly
        remark_lines = self.get_remark(350)
        if remark_lines is None:
            raise InvalidFileError(
                "File does not contain assembly information (REMARK 350)"
            )
        # Get lines corresponding to selected assembly ID
        assembly_start_i = None
        assembly_stop_i = None
        for i, line in enumerate(remark_lines):
            if line.startswith("BIOMOLECULE"):
                current_assembly_id = line[12:].strip()
                if assembly_start_i is not None:
                    # Start was already found -> this is the next entry
                    # -> this is the stop
                    assembly_stop_i = i
                    break
                if current_assembly_id == assembly_id or assembly_id is None:
                    assembly_start_i = i
        # In case of the final assembly of the file,
        # the 'stop' is the end of REMARK 350 lines
        assembly_stop_i = len(remark_lines) if assembly_stop_i is None else i
        if assembly_start_i is None:
            if assembly_id is None:
                raise InvalidFileError(
                    "File does not contain transformation expressions for assemblies"
                )
            else:
                raise KeyError(f"The assembly ID '{assembly_id}' is not found")
        assembly_lines = remark_lines[assembly_start_i:assembly_stop_i]

        # Get transformations for a set of chains
        chain_set_start_indices = [
            i
            for i, line in enumerate(assembly_lines)
            if line.startswith("APPLY THE FOLLOWING TO CHAINS")
        ]
        # Add exclusive stop at end of records
        chain_set_start_indices.append(len(assembly_lines))
        assembly = None
        for i in range(len(chain_set_start_indices) - 1):
            start = chain_set_start_indices[i]
            stop = chain_set_start_indices[i + 1]
            # Read affected chain IDs from the following line(s)
            affected_chain_ids = []
            transform_start = None
            for j, line in enumerate(assembly_lines[start:stop]):
                if any(
                    line.startswith(chain_signal_string)
                    for chain_signal_string in [
                        "APPLY THE FOLLOWING TO CHAINS:",
                        "                   AND CHAINS:",
                    ]
                ):
                    affected_chain_ids += [
                        chain_id.strip() for chain_id in line[30:].split(",")
                    ]
                else:
                    # Chain specification has finished
                    # BIOMT lines start directly after chain specification
                    transform_start = start + j
                    break
            # Parse transformations from BIOMT lines
            if transform_start is None:
                raise InvalidFileError("No 'BIOMT' records found for chosen assembly")
            rotations, translations = _parse_transformations(
                assembly_lines[transform_start:stop]
            )
            # Filter affected chains
            sub_structure = structure[
                ..., np.isin(structure.chain_id, affected_chain_ids)
            ]
            sub_assembly = _apply_transformations(
                sub_structure, rotations, translations
            )
            # Merge the chains with IDs for this transformation
            # with chains from other transformations
            if assembly is None:
                assembly = sub_assembly
            else:
                assembly += sub_assembly

        return assembly

    def get_unit_cell(
        self, model=None, altloc="first", extra_fields=[], include_bonds=False
    ):
        """
        Build a structure model containing all symmetric copies
        of the structure within a single unit cell, given by the space
        group.

        This function receives the data from ``REMARK 290`` records in
        the file.
        Consequently, this remark must be present in the file, which is
        usually only true for crystal structures.

        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model instead of the first model.
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
            Bonds, whose order could not be determined from the
            *Chemical Component Dictionary*
            (e.g. especially inter-residue bonds),
            have :attr:`BondType.ANY`, since the PDB format itself does
            not support bond orders.

        Returns
        -------
        symmetry_mates : AtomArray or AtomArrayStack
            All atoms within a single unit cell.
            The return type depends on the `model` parameter.

        Notes
        -----
        To expand the structure beyond a single unit cell, use
        :func:`repeat_box()` with the return value as its
        input.

        Examples
        --------

        >>> import os.path
        >>> file = PDBFile.read(os.path.join(path_to_structures, "1aki.pdb"))
        >>> atoms_in_unit_cell = file.get_unit_cell(model=1)
        """
        # Get base structure
        structure = self.get_structure(
            model,
            altloc,
            extra_fields,
            include_bonds,
        )
        # Get lines containing transformations for crystallographic symmetry
        remark_lines = self.get_remark(290)
        if remark_lines is None:
            raise InvalidFileError(
                "File does not contain crystallographic symmetry "
                "information (REMARK 350)"
            )
        transform_lines = [line for line in remark_lines if line.startswith("  SMTRY")]
        rotations, translations = _parse_transformations(transform_lines)
        return _apply_transformations(structure, rotations, translations)

    def get_symmetry_mates(
        self, model=None, altloc="first", extra_fields=[], include_bonds=False
    ):
        """
        Build a structure model containing all symmetric copies
        of the structure within a single unit cell, given by the space
        group.

        This function receives the data from ``REMARK 290`` records in
        the file.
        Consequently, this remark must be present in the file, which is
        usually only true for crystal structures.

        DEPRECATED: Use :meth:`get_unit_cell()` instead.

        Parameters
        ----------
        model : int, optional
            If this parameter is given, the function will return an
            :class:`AtomArray` from the atoms corresponding to the given
            model number (starting at 1).
            Negative values are used to index models starting from the
            last model instead of the first model.
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
            Bonds, whose order could not be determined from the
            *Chemical Component Dictionary*
            (e.g. especially inter-residue bonds),
            have :attr:`BondType.ANY`, since the PDB format itself does
            not support bond orders.

        Returns
        -------
        symmetry_mates : AtomArray or AtomArrayStack
            All atoms within a single unit cell.
            The return type depends on the `model` parameter.

        Notes
        -----
        To expand the structure beyond a single unit cell, use
        :func:`repeat_box()` with the return value as its
        input.

        Examples
        --------

        >>> import os.path
        >>> file = PDBFile.read(os.path.join(path_to_structures, "1aki.pdb"))
        >>> atoms_in_unit_cell = file.get_symmetry_mates(model=1)
        """
        warnings.warn(
            "'get_symmetry_mates()' is deprecated, use 'get_unit_cell()' instead",
            DeprecationWarning,
        )
        return self.get_unit_cell(model, altloc, extra_fields, include_bonds)

    def _index_models_and_atoms(self):
        # Line indices where a new model starts
        self._model_start_i = np.array(
            [i for i in range(len(self.lines)) if self.lines[i].startswith(("MODEL"))],
            dtype=int,
        )
        if len(self._model_start_i) == 0:
            # It could be an empty file or a file with a single model,
            # where the 'MODEL' line is missing
            for line in self.lines:
                if line.startswith(("ATOM", "HETATM")):
                    # Single model
                    self._model_start_i = np.array([0])
                    break

        # Line indices with ATOM or HETATM records
        self._atom_line_i = np.array(
            [
                i
                for i in range(len(self.lines))
                if self.lines[i].startswith(("ATOM", "HETATM"))
            ],
            dtype=int,
        )

    def _get_atom_record_indices_for_model(self, model):
        last_model = len(self._model_start_i)
        if model == 0:
            raise ValueError("The model index must not be 0")
        # Negative models mean index starting from last model
        model = last_model + model + 1 if model < 0 else model

        if model < last_model:
            line_filter = (self._atom_line_i >= self._model_start_i[model - 1]) & (
                self._atom_line_i < self._model_start_i[model]
            )
        elif model == last_model:
            line_filter = self._atom_line_i >= self._model_start_i[model - 1]
        else:
            raise ValueError(
                f"The file has {last_model} models, "
                f"the given model {model} does not exist"
            )
        return self._atom_line_i[line_filter]

    def _get_model_length(self):
        """
        Determine length of models and check that all models
        have equal length.
        """
        n_models = len(self._model_start_i)
        length = None
        for model_i in range(len(self._model_start_i)):
            model_start = self._model_start_i[model_i]
            model_stop = (
                self._model_start_i[model_i + 1]
                if model_i + 1 < n_models
                else len(self.lines)
            )
            model_length = np.count_nonzero(
                (self._atom_line_i >= model_start) & (self._atom_line_i < model_stop)
            )
            if length is None:
                length = model_length
            if model_length != length:
                raise InvalidFileError(
                    f"Model {model_i + 1} has {model_length} atoms, "
                    f"but model 1 has {length} atoms, must be equal"
                )
        return length

    def _get_bonds(self, atom_ids):
        conect_lines = [line for line in self.lines if line.startswith("CONECT")]

        # Mapping from atom ids to indices in an AtomArray
        atom_id_to_index = np.zeros(atom_ids[-1] + 1, dtype=int)
        try:
            for i, id in enumerate(atom_ids):
                atom_id_to_index[id] = i
        except IndexError as e:
            raise InvalidFileError("Atom IDs are not strictly increasing") from e

        bonds = []
        for line in conect_lines:
            center_id = atom_id_to_index[decode_hybrid36(line[6:11])]
            for i in range(11, 31, 5):
                id_string = line[i : i + 5]
                try:
                    id = atom_id_to_index[decode_hybrid36(id_string)]
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
                    line = f"CONECT{atom_ids[center_i]:>5}"
                line += f"{atom_ids[bonded_i]:>5}"
                n_added += 1
                if n_added == 4:
                    # Only a maximum of 4 bond partners can be put
                    # into a single line
                    # If there are more, use an extra record
                    n_added = 0
                    self.lines.append(line)
            if n_added > 0:
                self.lines.append(line)


def _parse_transformations(lines):
    """
    Parse the rotation and translation transformations from
    *REMARK* 290 or 350.
    Return as array of matrices and vectors respectively
    """
    # Each transformation requires 3 lines for the (x,y,z) components
    if len(lines) % 3 != 0:
        raise InvalidFileError("Invalid number of transformation vectors")
    n_transformations = len(lines) // 3

    rotations = np.zeros((n_transformations, 3, 3), dtype=float)
    translations = np.zeros((n_transformations, 3), dtype=float)

    transformation_i = 0
    component_i = 0
    for line in lines:
        # The first two elements (component and
        # transformation index) are not used
        transformations = [float(e) for e in line.split()[2:]]
        if len(transformations) != 4:
            raise InvalidFileError("Invalid number of transformation vector elements")
        rotations[transformation_i, component_i, :] = transformations[:3]
        translations[transformation_i, component_i] = transformations[3]

        component_i += 1
        if component_i == 3:
            # All (x,y,z) components were parsed
            # -> head to the next transformation
            transformation_i += 1
            component_i = 0

    return rotations, translations


def _apply_transformations(structure, rotations, translations):
    """
    Get subassembly by applying the given transformations to the input
    structure containing affected chains.
    """
    # Additional first dimension for 'structure.repeat()'
    assembly_coord = np.zeros((len(rotations),) + structure.coord.shape)

    # Apply corresponding transformation for each copy in the assembly
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        coord = structure.coord
        # Rotate
        coord = matrix_rotate(coord, rotation)
        # Translate
        coord += translation
        assembly_coord[i] = coord

    assembly = repeat(structure, assembly_coord)
    assembly.set_annotation(
        "sym_id", np.repeat(np.arange(len(rotations)), structure.array_length())
    )
    return assembly


def _check_pdb_compatibility(array, hybrid36):
    annot_categories = array.get_annotation_categories()

    if hybrid36:
        max_atoms = max_hybrid36_number(5)
        max_residues = max_hybrid36_number(4)
    else:
        max_atoms, max_residues = _PDB_MAX_ATOMS, _PDB_MAX_RESIDUES
    if "atom_id" in annot_categories:
        max_atom_id = np.max(array.atom_id)
    else:
        max_atom_id = array.array_length()

    if max_atom_id > max_atoms:
        warnings.warn(f"Atom IDs exceed {max_atoms:,}, will be wrapped")
    if (array.res_id > max_residues).any():
        warnings.warn(f"Residue IDs exceed {max_residues:,}, will be wrapped")
    if np.isnan(array.coord).any():
        raise BadStructureError("Coordinates contain 'NaN' values")
    if any([len(name) > 1 for name in array.chain_id]):
        raise BadStructureError("Some chain IDs exceed 1 character")
    if any([len(name) > 3 for name in array.res_name]):
        raise BadStructureError("Some residue names exceed 3 characters")
    if any([len(name) > 4 for name in array.atom_name]):
        raise BadStructureError("Some atom names exceed 4 characters")
    for i, coord_name in enumerate(["x", "y", "z"]):
        n_coord_digits = number_of_integer_digits(array.coord[..., i])
        if n_coord_digits > 4:
            raise BadStructureError(
                f"4 pre-decimal columns for {coord_name}-coordinates are "
                f"available, but array would require {n_coord_digits}"
            )
    if "b_factor" in annot_categories:
        n_b_factor_digits = number_of_integer_digits(array.b_factor)
        if n_b_factor_digits > 3:
            raise BadStructureError(
                "3 pre-decimal columns for B-factor are available, "
                f"but array would require {n_b_factor_digits}"
            )
    if "occupancy" in annot_categories:
        n_occupancy_digits = number_of_integer_digits(array.occupancy)
        if n_occupancy_digits > 3:
            raise BadStructureError(
                "3 pre-decimal columns for occupancy are available, "
                f"but array would require {n_occupancy_digits}"
            )
    if "charge" in annot_categories:
        # The sign can be omitted is it is put into the adjacent column
        n_charge_digits = number_of_integer_digits(np.abs(array.charge))
        if n_charge_digits > 1:
            raise BadStructureError(
                "1 column for charge is available, "
                f"but array would require {n_charge_digits}"
            )
