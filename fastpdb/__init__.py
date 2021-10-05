# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "fastpdb"
__author__ = "Patrick Kunzmann"
__all__ = ["PDBFile"]

import numpy as np
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from .fastpdb import PDBFile as RustPDBFile


class PDBFile(biotite.TextFile):

    def __init__(self):
        super().__init__()
        self._pdb_file = RustPDBFile([])
    
    @classmethod
    def read(cls, file):
        file = super().read(file)
        file._pdb_file = RustPDBFile(file.lines)
        return file
    
    def get_model_count(self):
        return self._pdb_file.get_model_count()

    def get_coord(self, model=None):
        if model is None:
            coord = self._pdb_file.parse_coord_multi_model()
        else:
            coord = self._pdb_file.parse_coord_single_model(model)
        return coord
    
    def get_structure(self, model=None, altloc="first", extra_fields=None, include_bonds=False):
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
        
        if isinstance(atoms, struc.AtomArray):
            self._pdb_file.write_single_model(
                atoms.coord, chain_id, atoms.res_id, ins_code,
                res_name, atoms.hetero, atom_name, element,
                atom_id, b_factor, occupancy, charge
            )
        else:
            raise TypeError(
                f"Expected AtomArray or AtomArrayStack, "
                f"but got {type(atoms).__name__}"
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
            atom_id = np.arange(1, atoms.array_length()+1) if atom_id is None else atom_id
            self._pdb_file.write_bonds(
                bonds, atom_id
            )

            
        self.lines = self._pdb_file.lines


# Copy docstrings
PDBFile.get_model_count.__doc__ = pdb.PDBFile.get_model_count.__doc__
PDBFile.get_coord.__doc__       = pdb.PDBFile.get_coord.__doc__
PDBFile.get_structure.__doc__   = pdb.PDBFile.get_structure.__doc__