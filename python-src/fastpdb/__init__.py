__name__ = "fastpdb"
__author__ = "Patrick Kunzmann"
__all__ = ["PDBFile"]
__version__ = "1.0.1"

import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from .fastpdb import PDBFile as RustPDBFile


class PDBFile(biotite.TextFile):
    r"""
    This class represents a PDB file.
    
    This class only provides support for reading/writing the pure atom
    information (``ATOM``, ``HETATM``, ``MODEL`` and ``ENDMDL``
    records).
    ``TER`` records cannot be written.
    
    See also
    --------
    PDBxFile
    
    Examples
    --------
    Load a ``\\*.pdb`` file, modify the structure and save the new
    structure into a new file:
    
    >>> import os.path
    >>> file = PDBFile.read(os.path.join(path_to_structures, "1l2y.pdb"))
    >>> array_stack = file.get_structure()
    >>> array_stack_mod = rotate(array_stack, [1,2,3])
    >>> file = PDBFile()
    >>> file.set_structure(array_stack_mod)
    >>> file.write(os.path.join(path_to_directory, "1l2y_mod.pdb"))
    """

    def __init__(self):
        super().__init__()
        self._pdb_file = RustPDBFile([])
    
    @classmethod
    def read(cls, file):
        file = super().read(file)
        file._pdb_file = RustPDBFile(file.lines)
        return file
    
    def get_model_count(self):
        """
        Get the number of models contained in the PDB file.

        Returns
        -------
        model_count : int
            The number of models.
        """
        return self._pdb_file.get_model_count()

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
            The coordinates read from the ``ATOM`` and ``HETATM``
            records of the file.
        
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
        res_name  = np.frombuffer(res_name,  dtype="U3")
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

        
        box = self._pdb_file.parse_box()
        if box is None:
            atoms.box = None
        else:
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
            bond_list = bond_list.merge(struc.connect_via_residue_names(
                atoms,
                # The information for non-hetero residues and water
                # are not part of CONECT records
                (~atoms.hetero) | struc.filter_solvent(atoms)
            ))
            # Remove bond order from inter residue bonds for consistency
            bond_list.remove_bond_order()
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
        chain_id  = np.frombuffer(atoms.chain_id,  dtype=np.uint32).reshape(-1, 4)
        ins_code  = np.frombuffer(atoms.ins_code,  dtype=np.uint32).reshape(-1, 1)
        res_name  = np.frombuffer(atoms.res_name,  dtype=np.uint32).reshape(-1, 3)
        atom_name = np.frombuffer(atoms.atom_name, dtype=np.uint32).reshape(-1, 6)
        element   = np.frombuffer(atoms.element,   dtype=np.uint32).reshape(-1, 2)
        
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

            
        self.lines = self._pdb_file.lines