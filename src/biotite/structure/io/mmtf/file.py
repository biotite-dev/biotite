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
    
    def get_structure(self, insertion_code=[], altloc=[],
                      model=None, extra_fields=[]):
        """
        Get an `AtomArray` or `AtomArrayStack` from the MMTF file.
        
        Parameters
        ----------
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
        model : int, optional
            If this parameter is given, the function will return an
            `AtomArray` from the atoms corresponding to the given model ID.
            If this parameter is omitted, an `AtomArrayStack` containing all
            models will be returned, even if the structure contains only one
            model.
        extra_fields : list of str, optional
            The strings in the list are optional annotation categories
            that should be stored in the uoput array or stack.
            There are 4 optional annotation identifiers:
            'atom_id', 'b_factor', 'occupancy' and 'charge'.
        
        Returns
        -------
        array : AtomArray or AtomArrayStack
            The return type depends on the `model` parameter.
        """
        dec = self.decoder
        if dec is None:
            raise ValueError("No structure is currently loaded")
        
        if model is None:
            length = _get_model_length(dec, 1)
            depth = dec.num_models
            # Check if each model has the same amount of atoms
            # If not, raise exception
            if length * dec.num_models != dec.num_atoms:
                raise BadStructureError("The models in the file have unequal "
                                        "amount of atoms, give an explicit "
                                        "model instead")
            array = AtomArrayStack(depth, length)
            array.coord = np.stack(
                [dec.x_coord_list,
                 dec.y_coord_list,
                 dec.z_coord_list],
                 axis=1
            ).reshape(depth, length, 3)
            # Create inscode and altloc arrays for the final filtering
            altloc_array = np.array(dec.alt_loc_list[:length], dtype="U1")
            inscode_array = np.zeros(length, dtype="U1")
            
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
            
            _fill_annotations(dec, 1, array, inscode_array, extra_charge)
        
        else:
            model_length = _get_model_length(dec, model)
            # Indices to filter coords and some annotations
            # for the specified model
            start_i = 0
            for m in range(1, model):
                start_i += _get_model_length(dec, m)
            stop_i = start_i + model_length
            array = AtomArray(model_length)
            array.coord[:,0] = dec.x_coord_list[start_i : stop_i]
            array.coord[:,1] = dec.y_coord_list[start_i : stop_i]
            array.coord[:,2] = dec.z_coord_list[start_i : stop_i]
            # Create inscode and altloc arrays for the final filtering
            altloc_array = np.array(dec.alt_loc_list[start_i : stop_i],
                                    dtype="U1")
            inscode_array = np.zeros(array.array_length(), dtype="U1")
            
            extra_charge = False
            if "charge" in extra_fields:
                extra_charge = True
                array.add_annotation("charge", int)
            if "atom_id" in extra_fields:
                array.set_annotation("atom_id",
                                     dec.atom_id_list[start_i : stop_i])
            if "b_factor" in extra_fields:
                array.set_annotation("b_factor",
                                     dec.b_factor_list[start_i : stop_i])
            if "occupancy" in extra_fields:
                array.set_annotation("occupancy",
                                     dec.occupancy_list[start_i : stop_i])
            
            _fill_annotations(dec, model, array, inscode_array, extra_charge)
        
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


def _get_model_length(decoder, model):
    atom_count = 0
    chain_i = 0
    res_i = 0
    for i in range(decoder.chains_per_model[model-1]):
        for j in range(decoder.groups_per_chain[chain_i]): 
            residue = decoder.group_list[decoder.group_type_list[res_i]]
            atom_count += len(residue["atomNameList"])
            res_i += 1
        chain_i += 1
    return atom_count

    
def _fill_annotations(decoder, model, array, inscode_array, extra_charge):
    chain_i = 0
    res_i = 0
    atom_i = 0
    non_hetero_list = ["L-PEPTIDE LINKING", "PEPTIDE LINKING",
                       "DNA LINKING", "RNA LINKING"]
    for i in range(decoder.chains_per_model[model-1]):
        chain_id = decoder.chain_name_list[chain_i]
        for j in range(decoder.groups_per_chain[chain_i]): 
            residue = decoder.group_list[decoder.group_type_list[res_i]]
            res_id = decoder.sequence_index_list[res_i] + 1
            res_name = residue["groupName"]
            hetero = (residue["chemCompType"] not in non_hetero_list)
            inscode = decoder.ins_code_list[res_i]
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
                
            