# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["get_sequence", "get_structure", "set_structure"]

import numpy as np
from ...error import BadStructureError
from ...atoms import Atom, AtomArray, AtomArrayStack
from ...filter import filter_inscode_and_altloc
from ....sequence.seqtypes import ProteinSequence
from collections import OrderedDict


def get_sequence(pdbx_file, data_block=None):
    """
    Get the protein sequences from the
    `entity_poly.pdbx_seq_one_letter_code_can` entry. 
    
    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    data_block : string, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
        
    Returns
    -------
    sequences : list of ProteinSequence
        The protein sequences for each entity (equivalent to chain in
        most cases).
    """
    poly_dict = pdbx_file.get_category("entity_poly", data_block)
    seq_string = poly_dict["pdbx_seq_one_letter_code_can"]
    sequences = []
    if isinstance(seq_string, np.ndarray):
        for string in seq_string:
            sequences.append(ProteinSequence(string))
    else:
        sequences.append(ProteinSequence(seq_string))
    return sequences


def get_structure(pdbx_file, model=None, data_block=None,
                  insertion_code=[], altloc=[], extra_fields=[]):
    """
    Create an `AtomArray` or `AtomArrayStack` from a `atom_site`
    category.
    
    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        `AtomArray` from the atoms corresponding to the given model ID.
        If this parameter is omitted, an `AtomArrayStack` containing all
        models will be returned, even if the structure contains only one
        model.
    data_block : string, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
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
    extra_fields : list of str, optional
        The strings in the list are entry names, that are
        additionally added as annotation arrays.
        The annotation category name will be the same as the PDBx
        subcategroy name. The array type is always `str`.
        There are 4 special field identifiers:
        'atom_id', 'b_factor', 'occupancy' and 'charge'.
        These will convert the respective subcategory into an
        annotation array with reasonable type.
        
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
        
    Examples
    --------

    >>> file = PDBxFile()
    >>> file.read("1l2y.cif")
    >>> arr = get_structure(file, model=1)
    >>> print(len(arr))
    304
    
    """
    atom_site_dict = pdbx_file.get_category("atom_site", data_block)
    models = atom_site_dict["pdbx_PDB_model_num"]
    if model is None:
        # For a stack, the annotation are derived from the first model
        model_dict = _get_model_dict(atom_site_dict, 1)
        model_count = int(models[-1])
        model_length = len(model_dict["group_PDB"])
        stack = AtomArrayStack(model_count, model_length)
        _fill_annotations(stack, model_dict, extra_fields)
        # Check if each model has the same amount of atoms
        # If not, raise exception
        atom_count = len(models)
        if model_length * model_count != atom_count:
            raise BadStructureError("The models in the file have unequal "
                                    "amount of atoms, give an explicit model "
                                    "instead")
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
        model_dict = _get_model_dict(atom_site_dict, model)
        model_length = len(model_dict["group_PDB"])
        array = AtomArray(model_length)
        _fill_annotations(array, model_dict, extra_fields)
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
                                      for e in model_dict["auth_seq_id"]]))
    array.set_annotation("res_name", model_dict["label_comp_id"].astype("U3"))
    array.set_annotation("hetero", (model_dict["group_PDB"] == "HETATM"))
    array.set_annotation("atom_name", model_dict["label_atom_id"].astype("U6"))
    array.set_annotation("element", model_dict["type_symbol"].astype("U2"))
    for field in extra_fields:
        if field == "atom_id":
            array.set_annotation("atom_id",
                                 model_dict["id"].astype(int))
        elif field == "b_factor":
            array.set_annotation("b_factor",
                                 model_dict["B_iso_or_equiv"].astype(float))
        elif field == "occupancy":
            array.set_annotation("occupancy",
                                 model_dict["occupancy"].astype(float))
        elif field == "charge":
            array.set_annotation("charge", np.array(
                [0 if charge in ["?","."] else int(charge)
                 for charge in model_dict["pdbx_formal_charge"]], dtype=int
            ))
        else:
            field_name = field[0]
            annot_name = field[1]
            array.set_annotation(annot_name,
                                 model_dict[field_name].astype(str))


def _filter_inscode_altloc(array, model_dict, inscode, altloc):
    try:
        inscode_array = model_dict["pdbx_PDB_ins_code"]
    except KeyError:
        # In case no insertion code column is existent
        inscode_array = None
    try:
        altloc_array = model_dict["label_alt_id"]
    except KeyError:
        # In case no insertion code column is existent
        altloc_array = None
    return array[..., filter_inscode_and_altloc(
        array, inscode, altloc, inscode_array, altloc_array
    )]


def _get_model_dict(atom_site_dict, model):
    model_dict = {}
    models = atom_site_dict["pdbx_PDB_model_num"]
    for key in atom_site_dict.keys():
        model_dict[key] = atom_site_dict[key][models == str(model)]
    return model_dict


def set_structure(pdbx_file, array, data_block=None):
    """
    Set the `atom_site` category with an
    `AtomArray` or `AtomArrayStack`.
    
    This will save the coordinates, the mandatory annotation categories
    and the optional annotation categories
    'atom_id', 'b_factor', 'occupancy' and 'charge'.
    If the array contains the annotation 'atom_id', these values will be
    used for atom numbering instead of continuous numbering.
    
    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    array : AtomArray or AtomArrayStack
        The structure to be written. If a stack is given, each array in
        the stack will be in a separate model.
    data_block : string, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
    
    Examples
    --------

    >>> file = PDBxFile()
    >>> set_structure(file, atom_array)
    >>> file.write("structure.cif")
    
    """
    # Fill PDBx columns from information
    # in structures' attribute arrays as good as possible
    # Use OrderedDict in order to ensure the usually used column order.
    atom_site_dict = OrderedDict()
    # Save list of annotation categories for checks,
    # if an optional category exists 
    annot_categories = array.get_annotation_categories()
    atom_site_dict["group_PDB"] = np.array(["ATOM" if e == False else "HETATM"
                                            for e in array.hetero])
    atom_site_dict["type_symbol"] = np.copy(array.element)
    atom_site_dict["label_atom_id"] = np.copy(array.atom_name)
    atom_site_dict["label_alt_id"] = np.full(array.array_length(), ".")
    atom_site_dict["label_comp_id"] = np.copy(array.res_name)
    atom_site_dict["label_asym_id"] = np.copy(array.chain_id)
    atom_site_dict["label_entity_id"] = _determine_entity_id(array.chain_id)
    atom_site_dict["label_seq_id"] = np.array(["." if e == -1 else str(e)
                                               for e in array.res_id])
    atom_site_dict["auth_seq_id"] = atom_site_dict["label_seq_id"]
    atom_site_dict["auth_comp_id"] = atom_site_dict["label_comp_id"]
    atom_site_dict["auth_asym_id"] = atom_site_dict["label_asym_id"]
    atom_site_dict["auth_atom_id"] = atom_site_dict["label_atom_id"]
    #Optional categories
    if "atom_id" in annot_categories:
        # Take values from 'atom_id' category
        atom_site_dict["id"] = array.atom_id.astype("U6")
    else:
        atom_site_dict["id"] = None
    if "b_factor" in annot_categories:
        atom_site_dict["B_iso_or_equiv"] = np.array(
            [f"{b:.2f}" for b in array.b_factor]
        )
    if "occupancy" in annot_categories:
        atom_site_dict["occupancy"] = np.array(
            [f"{occ:.2f}" for occ in array.occupancy]
        )
    if "charge" in annot_categories:
        atom_site_dict["pdbx_formal_charge"] = np.array(
            [f"{c:+d}" if c != 0 else "?" for c in array.charge]
        )
    # In case of a single model handle each coordinate
    # simply like a flattened array
    if (  type(array) == AtomArray or
         (type(array) == AtomArrayStack and array.stack_depth() == 1)  ):
        # 'ravel' flattens coord without copy
        # in case of stack with stack_depth = 1
        atom_site_dict["Cartn_x"] = np.array(
            [f"{c:.3f}" for c in np.ravel(array.coord[...,0])]
        )
        atom_site_dict["Cartn_y"] = np.array(
            [f"{c:.3f}" for c in np.ravel(array.coord[...,1])]
        )
        atom_site_dict["Cartn_z"] = np.array(
            [f"{c:.3f}" for c in np.ravel(array.coord[...,2])]
        )
        atom_site_dict["pdbx_PDB_model_num"] = np.full(
            array.array_length(), "1"
        )
    # In case of multiple models repeat annotations
    # and use model specific coordinates
    elif type(array) == AtomArrayStack:
        for key, value in atom_site_dict.items():
            atom_site_dict[key] = np.tile(value, reps=array.stack_depth())
        coord = np.reshape(array.coord,
                           (array.stack_depth()*array.array_length(), 3))
        atom_site_dict["Cartn_x"] = np.array(
            [f"{c:.3f}" for c in coord[:,0]]
        )
        atom_site_dict["Cartn_y"] = np.array(
            [f"{c:.3f}" for c in coord[:,1]]
        )
        atom_site_dict["Cartn_z"] = np.array(
            [f"{c:.3f}" for c in coord[:,2]]
        )
        models = np.repeat(
            np.arange(1, array.stack_depth()+1).astype(str),
            repeats=array.array_length()
        )
        atom_site_dict["pdbx_PDB_model_num"] = models
    else:
        raise ValueError("Structure must be AtomArray or AtomArrayStack")
    if not "atom_id" in annot_categories:
        # Count from 1
        atom_site_dict["id"] = (np.arange(1,len(atom_site_dict["group_PDB"])+1)
                               .astype("U6"))
    if data_block is None:
        data_blocks = pdbx_file.get_block_names()
        if len(data_blocks) == 0:
            raise TypeError("No data block is existent in PDBx file, "
                            "must be specified")
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
