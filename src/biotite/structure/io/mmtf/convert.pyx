# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

cimport cython
cimport numpy as np

import numpy as np
from .file import MMTFFile
from ...atoms import Atom, AtomArray, AtomArrayStack
from ...error import BadStructureError
from ...filter import filter_inscode_and_altloc

ctypedef np.int8_t int8
ctypedef np.int16_t int16
ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32

__all__ = ["get_structure", "set_structure"]

    
def get_structure(file, insertion_code=[], altloc=[],
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
        that should be stored in the output array or stack.
        There are 4 optional annotation identifiers:
        'atom_id', 'b_factor', 'occupancy' and 'charge'.
    
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    
    Examples
    --------

    >>> file = MMTFFile()
    >>> file.read(path)
    >>> array = get_structure(file, model=1)
    >>> print(array.array_length())
    >>> stack = get_structure(file)
    304
    >>> print(stack.stack_depth(), stack.array_length())
    38 304
    """
    cdef int i, j, m
    
    # Obtain (and potentially decode) required arrays/values from file
    cdef int atom_count = file["numAtoms"]
    cdef int model_count = file["numModels"]
    cdef np.ndarray chain_names = file["chainNameList"]
    cdef int32[:] chains_per_model = np.array(file["chainsPerModel"], np.int32)
    cdef int32[:] res_per_chain = np.array(file["groupsPerChain"], np.int32)
    cdef int32[:] res_type_i = file["groupTypeList"]
    cdef int32[:] res_ids = np.add(file["sequenceIndexList"],1, dtype=np.int32)
    cdef np.ndarray x_coord = file["xCoordList"]
    cdef np.ndarray y_coord = file["yCoordList"]
    cdef np.ndarray z_coord = file["zCoordList"]
    cdef np.ndarray b_factor = file["bFactorList"]
    cdef np.ndarray occupancy = file["occupancyList"]
    cdef np.ndarray altloc_all = file["altLocList"]
    cdef np.ndarray inscode = file["insCodeList"]
    cdef np.ndarray atom_ids = file["atomIdList"]
    
    # Create arrays from 'groupList' list of dictionaries
    cdef list group_list = file["groupList"]
    cdef list non_hetero_list = ["L-PEPTIDE LINKING", "PEPTIDE LINKING",
                                 "DNA LINKING", "RNA LINKING"]
    # Determine per-residue-count and maximum count
    # of atoms in each residue
    cdef np.ndarray atoms_per_res = np.zeros(len(group_list), dtype=np.int32)
    for i in range(len(group_list)):
        atoms_per_res[i] = len(group_list[i]["atomNameList"])
    cdef int32 max_atoms_per_res = np.max(atoms_per_res)
    #Create the arrays
    cdef np.ndarray res_names = np.zeros(len(group_list), dtype="U3")
    cdef np.ndarray hetero_res = np.zeros(len(group_list), dtype=np.bool)
    cdef np.ndarray atom_names = np.zeros((len(group_list), max_atoms_per_res),
                                          dtype="U6")
    cdef np.ndarray elements = np.zeros((len(group_list), max_atoms_per_res),
                                        dtype="U2")
    cdef np.ndarray charges = np.zeros((len(group_list), max_atoms_per_res),
                                          dtype=np.int32)
    # Fill the arrays
    for i in range(len(group_list)):
        residue = group_list[i]
        res_names[i] = residue["groupName"]
        hetero_res[i] = (residue["chemCompType"] not in non_hetero_list)
        atom_names[i, :atoms_per_res[i]] = residue["atomNameList"]
        elements[i, :atoms_per_res[i]] = residue["elementList"]
        charges[i, :atoms_per_res[i]] = residue["formalChargeList"]
    
    # Create the atom array (stack)
    cdef int depth, length
    cdef int start_i, stop_i
    cdef bint extra_charge
    cdef np.ndarray altloc_array
    cdef np.ndarray inscode_array
    if model == None:
        length = _get_model_length(1, res_type_i, chains_per_model,
                                   res_per_chain, atoms_per_res)
        depth = model_count
        # Check if each model has the same amount of atoms
        # If not, raise exception
        if length * model_count != atom_count:
            raise BadStructureError("The models in the file have unequal "
                                    "amount of atoms, give an explicit "
                                    "model instead")
        array = AtomArrayStack(depth, length)
        array.coord = np.stack(
            [x_coord,
             y_coord,
             z_coord],
             axis=1
        ).reshape(depth, length, 3)
        # Create inscode and altloc arrays for the final filtering
        altloc_array = altloc_all[:length]
        inscode_array = np.zeros(length, dtype="U1")
        
        extra_charge = False
        if "charge" in extra_fields:
            extra_charge = True
            array.add_annotation("charge", int)
        if "atom_id" in extra_fields:
            array.set_annotation("atom_id", atom_ids[:length])
        if "b_factor" in extra_fields:
            array.set_annotation("b_factor", b_factor[:length])
        if "occupancy" in extra_fields:
            array.set_annotation("occupancy", occupancy[:length])
        
        _fill_annotations(1, array, inscode, inscode_array, extra_charge,
                          chain_names, chains_per_model, res_per_chain,
                          res_type_i, res_ids, atoms_per_res, res_names,
                          hetero_res, atom_names, elements, charges)
    
    else:
        length = _get_model_length(model, res_type_i, chains_per_model,
                                   res_per_chain, atoms_per_res)
        # Indices to filter coords and some annotations
        # for the specified model
        start_i = 0
        for m in range(1, model):
            start_i += _get_model_length(m, res_type_i, chains_per_model,
                                         res_per_chain, atoms_per_res)
        stop_i = start_i + length
        array = AtomArray(length)
        array.coord[:,0] = x_coord[start_i : stop_i]
        array.coord[:,1] = y_coord[start_i : stop_i]
        array.coord[:,2] = z_coord[start_i : stop_i]
        # Create inscode and altloc arrays for the final filtering
        altloc_array = np.array(altloc_all[start_i : stop_i],
                                dtype="U1")
        inscode_array = np.zeros(array.array_length(), dtype="U1")
        
        extra_charge = False
        if "charge" in extra_fields:
            extra_charge = True
            array.add_annotation("charge", int)
        if "atom_id" in extra_fields:
            array.set_annotation("atom_id", atom_ids[start_i : stop_i])
        if "b_factor" in extra_fields:
            array.set_annotation("b_factor", b_factor[start_i : stop_i])
        if "occupancy" in extra_fields:
            array.set_annotation("occupancy", occupancy[start_i : stop_i])
        
        _fill_annotations(model, array, inscode, inscode_array, extra_charge,
                          chain_names, chains_per_model, res_per_chain,
                          res_type_i, res_ids, atoms_per_res, res_names,
                          hetero_res, atom_names, elements, charges)
    
    # Filter inscode and altloc and return
    # Format arrays for filter function
    altloc_array[altloc_array == ""] = " "
    inscode_array[inscode_array == ""] = " "
    return array[..., filter_inscode_and_altloc(
        array, insertion_code, altloc, inscode_array, altloc_array
    )]


def _get_model_length(int model, int32[:] res_type_i,
                      int32[:] chains_per_model,
                      int32[:] res_per_chain,
                      int32[:] atoms_per_res):
    cdef int atom_count = 0
    cdef int chain_i = 0
    cdef int res_i = 0
    cdef int i,j
    for i in range(chains_per_model[model-1]):
        for j in range(res_per_chain[chain_i]): 
            atom_count += atoms_per_res[res_type_i[res_i]]
            res_i += 1
        chain_i += 1
    return atom_count

    
def _fill_annotations(int model, array,
                      np.ndarray res_inscodes, np.ndarray atom_inscodes,
                      bint extra_charge, np.ndarray chain_names,
                      int32[:] chains_per_model, int32[:] res_per_chain,
                      int32[:] res_type_i, int32[:] res_ids,
                      np.ndarray atoms_per_res,
                      np.ndarray res_names, np.ndarray hetero_res,
                      np.ndarray atom_names, np.ndarray elements,
                      np.ndarray charges):
    # Get annotation arrays from atom array (stack)
    cdef np.ndarray chain_id  = array.chain_id
    cdef np.ndarray res_id    = array.res_id
    cdef np.ndarray res_name  = array.res_name
    cdef np.ndarray hetero    = array.hetero
    cdef np.ndarray atom_name = array.atom_name
    cdef np.ndarray element   = array.element
    cdef np.ndarray charge
    if extra_charge:
        charge = array.charge
    
    cdef chain_id_for_chain
    cdef res_name_for_res
    cdef inscode_for_res
    cdef bint hetero_for_res
    cdef int res_id_for_res
    cdef int type_i
    cdef int chain_i = 0
    cdef int res_i = 0
    cdef int atom_i = 0
    cdef int i, j, k
    for i in range(chains_per_model[model-1]):
        chain_id_for_chain = chain_names[chain_i]
        for j in range(res_per_chain[chain_i]): 
            res_id_for_res = res_ids[res_i]
            inscode_for_res = res_inscodes[res_i]
            type_i = res_type_i[res_i]
            res_name_for_res = res_names[type_i]
            hetero_for_res = hetero_res[type_i]
            for k in range(atoms_per_res[type_i]):
                chain_id[atom_i] = chain_id_for_chain
                res_id[atom_i]    = res_id_for_res
                hetero[atom_i]    = hetero_for_res
                res_name[atom_i]  = res_name_for_res
                atom_name[atom_i] = atom_names[type_i][k]
                element[atom_i]   = elements[type_i][k].upper()
                if extra_charge:
                    charge[atom_i] = charges[type_i][k]
                atom_inscodes[atom_i] = inscode_for_res
                atom_i += 1
            res_i += 1
        chain_i += 1
                

def set_structure(file, array):
    raise NotImplementedError()