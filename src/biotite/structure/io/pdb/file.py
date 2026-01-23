# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdb"
__author__ = "Patrick Kunzmann, Daniel Bauer, Claude J. Rogers"
__all__ = ["PDBFile"]


import copy
import itertools
import warnings
from collections import namedtuple
import numpy as np
import biotite.structure as struc
from biotite.file import InvalidFileError
from biotite.rust.structure.io.pdb import PDBFile as RustPDBFile
from biotite.rust.structure.io.pdb import (
    max_hybrid36_number,
)
from biotite.structure.atoms import repeat
from biotite.structure.bonds import BondList, connect_via_residue_names
from biotite.structure.error import BadStructureError
from biotite.structure.filter import (
    filter_solvent,
)
from biotite.structure.info.bonds import bonds_in_residue
from biotite.structure.io.util import number_of_integer_digits
from biotite.structure.util import matrix_rotate

_PDB_MAX_ATOMS = 99999
_PDB_MAX_RESIDUES = 9999

# slice objects for readability
_SPACE = slice(55, 66)
_Z = slice(66, 70)


class PDBFile(RustPDBFile):
    """
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

    def get_structure(
        self, model=None, altloc="first", extra_fields=None, include_bonds=False
    ):
        atoms = super().get_structure(model, altloc, extra_fields, include_bonds)

        # Replace empty strings for elements with guessed types
        # This is used e.g. for PDB files created by Gromacs
        empty_element_mask = atoms.element == ""
        if empty_element_mask.any():
            warnings.warn(
                f"{np.count_nonzero(empty_element_mask)} elements "
                "were guessed from atom name"
            )
            atoms.element[empty_element_mask] = struc.infer_elements(
                atoms.atom_name[empty_element_mask]
            )

        if include_bonds:
            # Add bonds inferred from CCD (only non-hetero residues + water)
            custom_bond_dict = {
                res_name: bonds_in_residue(res_name)
                for res_name in itertools.chain(
                    np.unique(atoms[..., ~atoms.hetero].res_name), ["HOH"]
                )
            }
            atoms.bonds = atoms.bonds.merge(
                connect_via_residue_names(atoms, custom_bond_dict=custom_bond_dict)
            )

        return atoms

    def set_structure(self, atoms, hybrid36=False):
        _check_pdb_compatibility(atoms, hybrid36)

        # PDB files only contains ``CONECT`` records for bonds between non-water hetero
        # residues and inter-residue bonds
        # -> Preprocess `AtomArray` to remove those bonds
        if atoms.bonds is not None:
            # We only replace the BondList -> shallow copy is enough
            atoms = copy.copy(atoms)
            hetero_indices = np.where(atoms.hetero & ~filter_solvent(atoms))[0]
            bond_array = atoms.bonds.as_array()
            bond_array = bond_array[
                np.isin(bond_array[:, 0], hetero_indices)
                | np.isin(bond_array[:, 1], hetero_indices)
                | (atoms.res_id[bond_array[:, 0]] != atoms.res_id[bond_array[:, 1]])
                | (atoms.chain_id[bond_array[:, 0]] != atoms.chain_id[bond_array[:, 1]])
            ]
            atoms.bonds = BondList(atoms.array_length(), bond_array)

        super().set_structure(atoms, hybrid36)

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
                    space_group = str(line[_SPACE])
                    z_val = int(line[_Z])
                except ValueError:
                    # File contains invalid 'CRYST1' record
                    raise InvalidFileError(
                        "File does not contain valid space group and/or Z values"
                    )
                break
        return SpaceGroupInfo(space_group=space_group, z_val=z_val)

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
                    self._set_line(
                        i, line[:55] + space_group_str + z_val_str + line[70:]
                    )
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
                [
                    line
                    for line in assembly_lines[transform_start:stop]
                    if len(line.strip()) > 0
                ]
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
