__name__ = "fastpdb"
__author__ = "Patrick Kunzmann"
__all__ = ["PDBFile"]
__version__ = "1.3.0"

import os
import warnings
import numpy as np
from biotite.file import is_text
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile as BiotitePDBFile
from .fastpdb import PDBFile as RustPDBFile


class PDBFile(BiotitePDBFile):

    def __init__(self):
        super().__init__()
        self._pdb_file = RustPDBFile([])

    @staticmethod
    def read(file):
        pdb_file = PDBFile()
        if isinstance(file, str):
            pdb_file._pdb_file = RustPDBFile.read(file)
        elif isinstance(file, bytes):
            pdb_file._pdb_file = RustPDBFile.read(file.decode("utf-8"))
        elif isinstance(file, os.PathLike):
            pdb_file._pdb_file = RustPDBFile.read(str(file))
        else:
            if not is_text(file):
                raise TypeError("A file opened in 'text' mode is required")
            pdb_file._pdb_file = RustPDBFile(file.read().splitlines())

        # Synchronize with PDB file representation in Rust
        pdb_file.lines = pdb_file._pdb_file.lines
        pdb_file._index_models_and_atoms()
        return pdb_file

    def get_model_count(self):
        return self._pdb_file.get_model_count()

    def get_remark(self, number):
        return self._pdb_file.parse_remark(int(number))

    def get_coord(self, model=None):
        if model is None:
            coord = self._pdb_file.parse_coord_multi_model()
        else:
            coord = self._pdb_file.parse_coord_single_model(model)
        return coord

    def get_structure(self, model=None, altloc="first", extra_fields=None, include_bonds=False):
        """
        Get an :class:`AtomArray` or :class:`AtomArrayStack` from the PDB file.

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
        if extra_fields is not None:
            include_atom_id   = "atom_id"   in extra_fields
            include_b_factor  = "b_factor"  in extra_fields
            include_occupancy = "occupancy" in extra_fields
            include_charge    = "charge"    in extra_fields
        else:
            include_atom_id   = False
            include_b_factor  = False
            include_occupancy = False
            include_charge    = False
        if include_bonds:
            # Required for mapping the bonded atom IDs to atom indices
            include_atom_id = True
        if altloc == "occupancy":
            include_occupancy = True


        if model is None:
            coord = self._pdb_file.parse_coord_multi_model()
            annotations = self._pdb_file.parse_annotations(
                1,
                include_atom_id, include_b_factor,
                include_occupancy, include_charge
            )
        else:
            coord = self._pdb_file.parse_coord_single_model(model)
            annotations = self._pdb_file.parse_annotations(
                model,
                include_atom_id, include_b_factor,
                include_occupancy, include_charge
            )
        (
            chain_id, res_id, ins_code, res_name,
            hetero, atom_name, element, altloc_id,
            atom_id, b_factor, occupancy, charge
        ) = annotations
        # Interpret uint32 arrays as unicode arrays
        chain_id  = np.frombuffer(chain_id,  dtype="U4")
        ins_code  = np.frombuffer(ins_code,  dtype="U1")
        res_name  = np.frombuffer(res_name,  dtype="U5")
        atom_name = np.frombuffer(atom_name, dtype="U6")
        element   = np.frombuffer(element,   dtype="U2")
        altloc_id = np.frombuffer(altloc_id, dtype="U1")

        if coord.ndim == 3:
            atoms = struc.AtomArrayStack(coord.shape[0], coord.shape[1])
            atoms.coord = coord
        else:
            atoms = struc.AtomArray(coord.shape[0])
            atoms.coord = coord

        atoms.chain_id  = chain_id
        atoms.res_id    = res_id
        atoms.ins_code  = ins_code
        atoms.res_name  = res_name
        atoms.hetero    = hetero
        atoms.atom_name = atom_name
        atoms.element   = element

        for field in (extra_fields if extra_fields is not None else []):
            if field == "atom_id":
                # Copy is necessary to avoid double masking in
                # later altloc ID filtering
                atoms.set_annotation("atom_id", atom_id.copy())
            elif field == "charge":
                atoms.set_annotation("charge", charge)
            elif field == "occupancy":
                atoms.set_annotation("occupancy", occupancy)
            elif field == "b_factor":
                atoms.set_annotation("b_factor", b_factor)
            else:
                raise ValueError(f"Unknown extra field: {field}")


        try:
            box = self._pdb_file.parse_box()
        except:
            warnings.warn(
                "File contains invalid 'CRYST1' record, box is ignored"
            )
        if box is not None:
            len_a, len_b, len_c, alpha, beta, gamma = box
            box = struc.vectors_from_unitcell(
                len_a, len_b, len_c,
                np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
            )
            if isinstance(atoms, struc.AtomArray):
                atoms.box = box
            else:
                atoms.box = np.repeat(
                    box[np.newaxis, ...], atoms.stack_depth(), axis=0
                )


        # Filter altloc IDs
        if altloc == "occupancy":
            filter = struc.filter_highest_occupancy_altloc(
                atoms, altloc_id, occupancy
            )
            atoms = atoms[..., filter]
            atom_id = atom_id[filter] if atom_id is not None else None
        elif altloc == "first":
            filter = struc.filter_first_altloc(atoms, altloc_id)
            atoms = atoms[..., filter]
            atom_id = atom_id[filter] if atom_id is not None else None
        elif altloc == "all":
            atoms.set_annotation("altloc_id", altloc_id)
        else:
            raise ValueError(f"'{altloc}' is not a valid 'altloc' option")


        if include_bonds:
            bond_list = struc.BondList(
                atoms.array_length(), self._pdb_file.parse_bonds(atom_id)
            )
            bond_list = bond_list.merge(struc.connect_via_residue_names(atoms))
            atoms.bonds = bond_list


        return atoms


    def set_structure(self, atoms):
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

        Notes
        -----
        If `array` has an associated :class:`BondList`, ``CONECT``
        records are also written for all non-water hetero residues
        and all inter-residue connections.
        """
        # Reset lines of text
        self._pdb_file = RustPDBFile([])


        # Write 'CRYST1' record
        if atoms.box is not None:
            box = atoms.box
            if box.ndim == 3:
                box = box[0]
            len_a, len_b, len_c, alpha, beta, gamma \
                = struc.unitcell_from_vectors(box)
            self._pdb_file.write_box(
                len_a, len_b, len_c,
                np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
            )


        # Write 'ATOM' and 'MODEL' records
        # Convert Unicode arrays into uint32 arrays for usage in Rust
        chain_id  = _convert_unicode_to_uint32(atoms.chain_id)
        ins_code  = _convert_unicode_to_uint32(atoms.ins_code)
        res_name  = _convert_unicode_to_uint32(atoms.res_name)
        atom_name = _convert_unicode_to_uint32(atoms.atom_name)
        element   = _convert_unicode_to_uint32(atoms.element)

        categories = atoms.get_annotation_categories()
        atom_id   = atoms.atom_id   if "atom_id"   in categories else None
        b_factor  = atoms.b_factor  if "b_factor"  in categories else None
        occupancy = atoms.occupancy if "occupancy" in categories else None
        charge    = atoms.charge    if "charge"    in categories else None

        # Convert to correct dtype for Rust function call, if necessary
        coord = atoms.coord.astype(np.float32, copy=False)
        res_id = atoms.res_id.astype(np.int64, copy=False)
        hetero = atoms.hetero.astype(bool, copy=False)
        if atom_id is not None:
            atom_id = atom_id.astype(np.int64, copy=False)
        if b_factor is not None:
            b_factor = b_factor.astype(np.float64, copy=False)
        if occupancy is not None:
            occupancy = occupancy.astype(np.float64, copy=False)
        if charge is not None:
            charge = charge.astype(np.int64, copy=False)

        # Treat a single model as multi-model structure
        if coord.ndim == 2:
            coord = coord[np.newaxis, :, :]

        self._pdb_file.write_models(
            coord, chain_id, res_id, ins_code,
            res_name, hetero, atom_name, element,
            atom_id, b_factor, occupancy, charge
        )

        # Write 'CONECT' records
        if atoms.bonds is not None:
            # Only non-water hetero records and connections between
            # residues are added to the records
            hetero_indices = np.where(atoms.hetero & ~struc.filter_solvent(atoms))[0]
            bond_array = atoms.bonds.as_array()
            bond_array = bond_array[
                np.isin(bond_array[:,0], hetero_indices) |
                np.isin(bond_array[:,1], hetero_indices) |
                (atoms.res_id  [bond_array[:,0]] != atoms.res_id  [bond_array[:,1]]) |
                (atoms.chain_id[bond_array[:,0]] != atoms.chain_id[bond_array[:,1]])
            ]
            # Bond type is unused since PDB does not support bond orders
            bonds, _ = struc.BondList(
                atoms.array_length(), bond_array
            ).get_all_bonds()
            atom_id = np.arange(1, atoms.array_length()+1, dtype=np.int64) \
                      if atom_id is None else atom_id
            self._pdb_file.write_bonds(
                bonds.astype(np.int32, copy=False), atom_id
            )

        # Synchronize with PDB file representation in Rust
        self.lines = self._pdb_file.lines
        self._index_models_and_atoms()


    def _index_models_and_atoms(self):
        self._pdb_file.index_models_and_atoms()
        self._model_start_i = self._pdb_file.model_start_i
        self._atom_line_i = self._pdb_file.atom_line_i


def _convert_unicode_to_uint32(array):
    """
    Convert a unicode string array into a 2D uint32 array.

    The second dimension corresponds to the character position within a
    string.
    """
    dtype = array.dtype
    if not np.issubdtype(dtype, np.str_):
        raise TypeError("Expected unicode string array")
    length = array.shape[0]
    n_char = dtype.itemsize // 4
    return np.frombuffer(array, dtype=np.uint32).reshape(length, n_char)