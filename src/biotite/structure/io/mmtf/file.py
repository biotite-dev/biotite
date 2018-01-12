# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import numpy as np
from ...atoms import Atom, AtomArray, AtomArrayStack
from ....file import File
from ...error import BadStructureError
from ...filter import filter_inscode_and_altloc
import mmtf

__all__ = ["MMTFFile"]


class MMTFFile(File):
    
    def __init__(self):
        self.decoder = None
        self.encoder = None
    
    def read(self, file_name):
        self.decoder = mmtf.parse(file_name)
    
    def write(self, file_name):
        raise NotImplementedError()
    
    def get_structure(self, extra_fields=[], insertion_code=[], altloc=[]):
        dec = self.decoder
        if dec is None:
            raise ValueError("No structure is currently loaded")
        if dec.num_models == 1:
            length = dec.num_atoms
            array = AtomArray(length)
            array.coord = np.stack(
                [dec.x_coord_list,
                 dec.y_coord_list,
                 dec.z_coord_list],
                 axis=1
            )
        else:
            length = dec.num_atoms // dec.num_models
            depth = dec.num_models
            array = AtomArrayStack(depth, length)
            array.coord = np.stack(
                [dec.x_coord_list,
                 dec.y_coord_list,
                 dec.z_coord_list],
                 axis=1
            ).reshape(depth, length, 3)
        # Create inscode and altloc arrays for the final filtering
        altloc_array = np.array(dec.alt_loc_list, dtype="U1")
        inscode_array = np.zeros(array.array_length(), dtype="U1")
        
        extra_charge = False
        if "charge" in extra_fields:
            extra_charge = True
            array.add_annotation("charge", int)
        if "atom_id" in extra_fields:
            array.set_annotation("atom_id", dec.atom_id_list[:length])
        if "b_factor" in extra_fields:
            array.set_annotation("b_factor", dec.b_factor_list[:length])
        if "occupancy" in extra_fields:
            array.set_annotation("occupancy", dec.occupancy_list[:length])
        
        chain_i = 0
        res_i = 0
        atom_i = 0
        for i in range(dec.chains_per_model[0]):
            chain_id = dec.chain_name_list[chain_i]
            for j in range(dec.groups_per_chain[chain_i]): 
                residue = dec.group_list[dec.group_type_list[res_i]]
                res_id = dec.sequence_index_list[res_i] + 1
                res_name = residue["groupName"]
                hetero = (residue["chemCompType"] == "NON-POLYMER")
                inscode = dec.ins_code_list[res_i]
                for k in range(len(residue["atomNameList"])):
                    array.chain_id[atom_i]  = chain_id
                    array.res_id[atom_i]    = res_id
                    array.hetero[atom_i]    = hetero
                    array.res_name[atom_i]  = res_name
                    array.atom_name[atom_i] = residue["atomNameList"][k]
                    array.element[atom_i]   = residue["elementList"][k].upper()
                    if extra_charge:
                        array.charge[atom_i] = residue["formalChargeList"][k]
                    inscode_array[atom_i] = inscode
                    atom_i += 1
                res_i += 1
            chain_i += 1
        
        # Filter inscode and altloc and return
        # Format arrays for filter function
        altloc_array[altloc_array == ""] = " "
        inscode_array[inscode_array == ""] = " "
        return array[..., filter_inscode_and_altloc(
            array, insertion_code, altloc, inscode_array, altloc_array
        )]
        
    def set_structure(self, array):
        raise NotImplementedError()
                
            