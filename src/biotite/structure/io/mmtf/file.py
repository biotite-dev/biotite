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
    """
    This class represents a MMTF file.
    
    This class only a parser for MMTF files. Writing MMTF files is not
    possible at this point.
    
    Examples
    --------
    
        >>> file = MMTFFile()
        >>> file.read("1l2y.mmtf")
        >>> array_stack = file.get_structure()
    
    """
    
    def __init__(self):
        self.decoder = None
        self.encoder = None
    
    def read(self, file_name):
        """
        Parse a MMTF file.
        
        Parameters
        ----------
        file_name : str
            The name of the file to be read.
        """
        self.decoder = mmtf.parse(file_name)
    
    def write(self, file_name):
        """
        Not implemented yet.        
        """
        raise NotImplementedError()
    
    def get_structure(self, extra_fields=[], insertion_code=[], altloc=[]):
        """
        Get an `AtomArray` or `AtomArrayStack` from the MMTF file.
        
        Parameters
        ----------
        extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the uoput array or stack.
            There are 4 optional annotation identifiers:
            'atom_id', 'b_factor', 'occupancy' and 'charge'.
        insertion_code : list of tuple, optional
            In case the structure contains insertion codes, those can be
            specified here: Each tuple consists of an integer, specifying
            the residue ID, and a letter, specifying the insertion code.
            By default no insertions are used.
        altloc : list of tuple, optional
            In case the structure contains *altloc* entries, those can be
            specified here: Each tuple consists of an integer, specifying
            the residue ID, and a letter, specifying the *altloc* ID.
            By default the location with the *altloc* ID "A" is used.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            A stack is returned, if this file contains multiple models,
            otherwise an array is returned.
        """
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
        altloc_array = np.array(dec.alt_loc_list[:length], dtype="U1")
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
        non_hetero_list = ["L-PEPTIDE LINKING", "PEPTIDE LINKING",
                           "DNA LINKING", "RNA LINKING"]
        for i in range(dec.chains_per_model[0]):
            chain_id = dec.chain_name_list[chain_i]
            for j in range(dec.groups_per_chain[chain_i]): 
                residue = dec.group_list[dec.group_type_list[res_i]]
                res_id = dec.sequence_index_list[res_i] + 1
                res_name = residue["groupName"]
                hetero = (residue["chemCompType"] not in non_hetero_list)
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
    
    def get_decoder(self):
        """
        Get the internally used `MMTFDecoder` from the package
        `mmtf-python`.
        
        Returns
        -------
        decoder : MMTFDecoder
            The decoder used for file parsing.
        """
        if self.decoder is None:
            raise ValueError("No structure is currently loaded")
        return self.decoder
        
    def set_structure(self, array):
        """
        Not implemented yet.        
        """
        raise NotImplementedError()
                
            