# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np
from ...error import BadStructureError
from ...atoms import Atom, AtomArray, AtomArrayStack
from collections import OrderedDict


def get_structure(pdbx_file, data_block=None, insertion_code=None,
                  altloc=None, model=None, extra_fields=None):
    atom_site_dict = pdbx_file.get_category("atom_site", data_block)
    models = atom_site_dict["pdbx_PDB_model_num"]
    if model is None:
        stack = AtomArrayStack()
        # For a stack, the annotation are derived from the first model
        model_dict = _get_model_dict(atom_site_dict, 1)
        _fill_annotations(stack, model_dict, extra_fields)
        model_count = int(models[-1])
        model_length = len(stack.chain_id)
        # Check if each model has the same amount of atoms
        # If not, raise exception
        if model_length * model_count != len(models):
            raise BadStructureError("The models in the file have unequal"
                            "amount of atoms, give an explicit model instead")
        stack.coord = np.zeros((model_count, model_length, 3), dtype=float)
        stack.coord[:,:,0] = atom_site_dict["Cartn_x"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,1] = atom_site_dict["Cartn_y"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,2] = atom_site_dict["Cartn_z"].reshape((model_count,
                                                                model_length))
        stack = _filter_inscode_altloc(stack, model_dict,
                                       insertion_code, altloc)
        return stack
    else:
        array = AtomArray()
        model_dict = _get_model_dict(atom_site_dict, model)
        _fill_annotations(array, model_dict, extra_fields)
        model_length = len(array.chain_id)
        model_filter = (models == str(model))
        array.coord = np.zeros((model_length, 3), dtype=float)
        array.coord[:,0]= atom_site_dict["Cartn_x"][model_filter].astype(float)
        array.coord[:,1]= atom_site_dict["Cartn_y"][model_filter].astype(float)
        array.coord[:,2]= atom_site_dict["Cartn_z"][model_filter].astype(float)
        array = _filter_inscode_altloc(array, model_dict,
                                       insertion_code, altloc)
        return array
        

def _fill_annotations(array, model_dict, extra_fields):
    array.set_annotation("chain_id", model_dict["auth_asym_id"].astype("U3"))
    array.set_annotation("res_id", np.array([-1 if e in [".","?"] else int(e)
                                             for e in model_dict["label_seq_id"]]))
    array.set_annotation("res_name", model_dict["label_comp_id"].astype("U3"))
    array.set_annotation("hetero", (model_dict["group_PDB"] == "HETATM"))
    array.set_annotation("atom_name", model_dict["label_atom_id"].astype("U6"))
    array.set_annotation("element", model_dict["type_symbol"].astype("U2"))
    if extra_fields is not None:
        for field in extra_fields:
            field_name = field[0]
            annot_name = field[1]
            array.set_annotation(annot_name, model_dict[field_name])


def _filter_inscode_altloc(array, model_dict, inscode, altloc):
    inscode_array = model_dict["pdbx_PDB_ins_code"]
    altloc_array = model_dict["label_alt_id"]
    # Default: Filter all atoms with insertion code ".", "?" or "A"
    inscode_filter = np.in1d(inscode_array, [".","?","A"],
                             assume_unique=True)
    # Now correct filter for every given insertion code
    if inscode is not None:
        for code in inscode:
            residue = code[0]
            insertion = code[1]
            residue_filter = (array.res_id == residue)
            # Reset filter for given res_id
            inscode_filter &= ~residue_filter
            # Choose atoms of res_id with insertion code
            inscode_filter |= residue_filter & (inscode_array == insertion)
    # Same with altlocs
    altloc_filter = np.in1d(altloc_array, [".","?","A"],
                            assume_unique=True)
    if altloc is not None:
        for loc in altloc:
            residue = loc[0]
            altloc = loc[1]
            residue_filter = (array.res_id == residue)
            altloc_filter &= ~residue_filter
            altloc_filter |= residue_filter & (altloc_array == altloc)
    # Apply combined filters
    return array[..., inscode_filter & altloc_filter]
    


def _get_model_dict(atom_site_dict, model):
    model_dict = {}
    models = atom_site_dict["pdbx_PDB_model_num"]
    for key in atom_site_dict.keys():
        model_dict[key] = atom_site_dict[key][models == str(model)]
    return model_dict


def set_structure(pdbx_file, array, data_block=None):
    """
    if type(array) == AtomArrayStack:
        models = array
    elif type(array) == AtomArray:
        models = [array]
    else raise ValueError("Structure must be AtomArray or AtomArrayStack")
    """
    atom_site_dict = OrderedDict()
    atom_site_dict["group_PDB"] = np.array(["ATOM" if e == False else "HETATM"
                                            for e in array.hetero])
    atom_site_dict["id"] = None
    atom_site_dict["type_symbol"] = np.copy(array.element)
    atom_site_dict["label_atom_id"] = np.copy(array.atom_name)
    atom_site_dict["label_alt_id"] = np.full(array.annotation_length(), ".")
    atom_site_dict["label_comp_id"] = np.copy(array.res_name)
    atom_site_dict["label_asym_id"] = np.copy(array.chain_id)
    atom_site_dict["label_entity_id"] = _determine_entity_id(array.chain_id)
    atom_site_dict["label_seq_id"] = np.array(["." if e == -1 else str(e)
                                            for e in array.res_id])
    atom_site_dict["auth_asym_id"] = np.copy(array.chain_id)
    if type(array) == AtomArrayStack:
        for key, value in atom_site_dict.items():
            atom_site_dict[key] = np.tile(value, reps=len(array))
        coord = np.reshape(array.coord,
                           (len(array)*array.annotation_length(), 3))
        atom_site_dict["Cartn_x"] = coord[:,0].astype(str)
        atom_site_dict["Cartn_y"] = coord[:,1].astype(str)
        atom_site_dict["Cartn_z"] = coord[:,2].astype(str)
        models = np.repeat(np.arange(1, len(coord)+1).astype(str),
                           repeats=array.annotation_length())
        atom_site_dict["pdbx_PDB_model_num"] = models
        atom_site_dict["id"] = (np.arange(1,array.annotation_length()+1)
                           .astype("U6"))
    elif type(array) == AtomArray:
        atom_site_dict["Cartn_x"] = array.coord[:,0].astype(str)
        atom_site_dict["Cartn_y"] = array.coord[:,1].astype(str)
        atom_site_dict["Cartn_z"] = array.coord[:,2].astype(str)
        atom_site_dict["pdbx_PDB_model_num"] = np.full(len(array), "1")
    else:
        raise ValueError("Structure must be AtomArray or AtomArrayStack")
    atom_site_dict["id"] = (np.arange(1,len(atom_site_dict["group_PDB"])+1)
                           .astype("U6"))
    if data_block is None:
        data_blocks = pdbx_file.get_block_names()
        if len(data_blocks) == 0:
            raise TypeError("No data block is existent in PDB file, must be specified")
        else:
            data_block = data_blocks[0]
    pdbx_file.set_category("atom_site", atom_site_dict, data_block)


def _determine_entity_id(chain_id):
    entity_id = np.zeros(len(chain_id), dtype=int)
    # Dictionary that translates chain_id to entity_id
    id_translation = {}
    id = 1
    for i in range(len(chain_id)):
        try:
            entity_id[i] = id_translation[chain_id[i]]
        except:
            # chain_id is not in dictionary -> new entry
            id_translation[chain_id[i]] = id
            entity_id[i] = id_translation[chain_id[i]]
            id += 1
    return entity_id.astype(str)
