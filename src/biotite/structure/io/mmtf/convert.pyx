# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

cimport cython
cimport numpy as np

import numpy as np
from .file import MMTFFile
from ...atoms import Atom, AtomArray, AtomArrayStack
from ...bonds import BondList
from ...error import BadStructureError
from ...filter import filter_inscode_and_altloc
from ...residues import get_residue_starts

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
                  model=None, extra_fields=[], include_bonds=False):
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
    cdef np.ndarray index_list = file["sequenceIndexList"]
    # Sequence index starts at 0, res IDs at 1
    # -> increment (the hetero residues (-1) exclusive)
    index_list[index_list != -1] += 1
    cdef int32[:] res_ids = index_list
    cdef np.ndarray x_coord = file["xCoordList"]
    cdef np.ndarray y_coord = file["yCoordList"]
    cdef np.ndarray z_coord = file["zCoordList"]
    cdef np.ndarray b_factor
    if "b_factor" in extra_fields:
        b_factor = file["bFactorList"]
    cdef np.ndarray occupancy
    if "occupancy" in extra_fields:
        occupancy = file["occupancyList"]
    cdef np.ndarray atom_ids
    if "atom_id" in extra_fields:
        atom_ids = file["atomIdList"]
    cdef np.ndarray altloc_all
    cdef np.ndarray inscode
    try:
        altloc_all = file["altLocList"]
    except KeyError:
        altloc_all = None
    try:
        inscode = file["insCodeList"]
    except KeyError:
        inscode = None
    
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
        if altloc_all is not None:
            altloc_array = altloc_all[:length]
        else:
            altloc_array = None
        if inscode is not None:
            inscode_array = np.zeros(length, dtype="U1")
        else:
            inscode_array = None
        
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
        
        if include_bonds:
            array.bonds = _create_bond_list(
                1, file["bondAtomList"], file["bondOrderList"],
                0, length, file["numAtoms"], group_list, res_type_i,
                atoms_per_res, res_per_chain, chains_per_model
            )
    
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
        if altloc_all is not None:
            altloc_array = np.array(altloc_all[start_i : stop_i], dtype="U1")
        else:
            altloc_array = None
        if inscode is not None:
            inscode_array = np.zeros(array.array_length(), dtype="U1")
        else:
            inscode_array = None
        
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
        if include_bonds:
            array.bonds = _create_bond_list(
                model, file["bondAtomList"], file["bondOrderList"],
                start_i, stop_i, file["numAtoms"], group_list, res_type_i,
                atoms_per_res, res_per_chain, chains_per_model
            )
    
    # Filter inscode and altloc and return
    # Format arrays for filter function
    if altloc_array is not None:
        altloc_array[altloc_array == ""] = " "
    if inscode_array is not None:
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
            if res_inscodes is not None:
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
                if res_inscodes is not None:
                    atom_inscodes[atom_i] = inscode_for_res
                atom_i += 1
            res_i += 1
        chain_i += 1


def _create_bond_list(int model, np.ndarray bonds, np.ndarray bond_types,
                      int model_start, int model_stop, int atom_count,
                      list group_list, int32[:] res_type_i,
                      int32[:] atoms_per_res,
                      int32[:] res_per_chain, int32[:] chains_per_model):
    cdef int i=0, j=0
    
    # Determine per-residue-count and maximum count
    # of bonds in each residue
    cdef int32[:] bonds_per_res = np.zeros(len(group_list), dtype=np.int32)
    for i in range(len(group_list)):
        bonds_per_res[i] = len(group_list[i]["bondOrderList"])
    cdef int32 max_bonds_per_res = np.max(bonds_per_res)

    # Create arrays for intra-residue bonds and bond types
    cdef np.ndarray intra_bonds = np.zeros(
        (len(group_list), max_bonds_per_res, 3), dtype=np.uint32
    )
    # Array for intermediate storing
    cdef np.ndarray bonds_in_residue
    # Dictionary for groupList entry
    cdef dict residue
    # Fill the array
    for i in range(len(group_list)):
        residue = group_list[i]
        bonds_in_residue = np.array(residue["bondAtomList"], dtype=np.uint32)
        intra_bonds[i, :bonds_per_res[i], :2] \
            = bonds_in_residue.reshape((len(bonds_in_residue)//2, 2))
        intra_bonds[i, :bonds_per_res[i], 2] = residue["bondOrderList"]

    # Unify intra-residue bonds to one BondList
    cdef int chain_i=0, res_i=0
    cdef int type_i
    intra_bond_list = BondList(0)
    for i in range(chains_per_model[model-1]):
        for j in range(res_per_chain[chain_i]): 
            type_i = res_type_i[res_i]
            bond_list_per_res = BondList(
                atoms_per_res[type_i],
                intra_bonds[type_i, :bonds_per_res[type_i]]
            )
            intra_bond_list += bond_list_per_res
            res_i += 1
        chain_i += 1
    
    # Add inter-residue bonds to BondList
    cdef np.ndarray inter_bonds = np.zeros((len(bond_types), 3),
                                           dtype=np.uint32)
    inter_bonds[:,:2] = bonds.reshape((len(bond_types), 2))
    inter_bonds[:,2] = bond_types
    inter_bond_list = BondList(atom_count, inter_bonds)
    inter_bond_list = inter_bond_list[model_start : model_stop]
    global_bond_list = BondList(
        model_stop - model_start,
        np.concatenate([inter_bond_list.as_array(),
                        intra_bond_list.as_array()], axis=0)
    )
    return global_bond_list
                

def set_structure(file, array, assume_unique=True):
    cdef int i=0, j=0
    cdef array_length = array.array_length()
    
    # Get annotation arrays from atom array (stack)
    cdef np.ndarray arr_chain_id  = array.chain_id
    cdef np.ndarray arr_res_id    = array.res_id
    cdef np.ndarray arr_res_name  = array.res_name
    cdef np.ndarray arr_hetero    = array.hetero
    cdef np.ndarray arr_atom_name = array.atom_name
    cdef np.ndarray arr_element   = array.element
    cdef np.ndarray arr_charge    = None
    if "charge" in array.get_annotation_categories():
        arr_charge = array.charge

    # Residue start indices
    # Since the stop of i is the start of i+1,
    # The exclusive end of the atom array is appended
    # to enable convenient usage in the following loops
    cdef np.ndarray starts = np.append(get_residue_starts(array),
                                       [array_length])

    ### Preparing the group list ###
    # List of residues used for setting the file's 'groupList'
    cdef list residues
    # An entry in 'residues'
    cdef dict curr_res
    # Stores a tuple of residue name and length for fast lookup in dict
    cdef tuple res_tuple
    # Dictionary with indices to list of residues as values
    cdef dict res_tuple_dict
    # Index to list of residues
    cdef int residue_i
    # List of indices to list of residues
    cdef np.ndarray res_types
    # Start and exclusive stop of on residue interval
    cdef int start
    cdef int stop
    # Amount of atoms in a residue
    cdef int res_length
    # Name of a residue
    cdef res_name
    if assume_unique:
        # 'len(starts)-1' since 'starts' has the end
        # of the atom array appended
        residue_i_array = np.zeros(len(starts)-1, dtype=np.int32)
        res_types = np.zeros(len(starts)-1, dtype=np.int32)
        residues = []
        res_tuple_dict = {}
        for i in range(len(starts)-1):
            start = starts[i]
            stop = starts[i+1]
            res_length = stop - start
            res_name = arr_res_name[start]
            res_tuple = (res_name, res_length)
            residue_i = res_tuple_dict.get(res_tuple, -1)
            if residue_i == -1:
                # New entry in dictionary
                curr_res = {}
                curr_res["atomNameList"] = arr_atom_name[start:stop].tolist()
                curr_res["elementList"] = arr_element[start:stop].tolist()
                if arr_charge is not None:
                    curr_res["formalChargeList"] \
                        = arr_charge[start:stop].tolist()
                else:
                    curr_res["formalChargeList"] = [0] * (stop-start)
                curr_res["groupName"] = res_name
                if arr_hetero[start]:
                    curr_res["chemCompType"] = "NON-POLYMER"
                else:
                    curr_res["chemCompType"] = "PEPTIDE LINKING"
                    # TODO: Differentiate cases of different polymers
                curr_res["bondAtomList"] = []
                curr_res["bondOrderList"] = []
                # Add new residue to list
                residue_i = len(residues)
                residues.append(curr_res)
                res_tuple_dict[res_tuple] = residue_i
                res_types[i] = residue_i
            else:
                # Put already known residue to residue types
                res_types[i] = residue_i
    else:
        raise NotImplementedError()
    
    ### Convert annotation arrays into MMTF arrays ###
    # Pessimistic assumption on length of arrays
    # -> At maximum as large as atom array
    cdef np.ndarray chain_names = np.zeros(array_length, dtype="U4")
    cdef np.ndarray res_per_chain = np.zeros(array_length, dtype=np.int32)
    # Variables for storing last and current chain  ID
    cdef last_chain_id = arr_chain_id[0]
    cdef curr_chain_id
    # Counter for chain length
    cdef int res_counter = 0
    i = 0
    j = 0
    for i in range(len(starts)-1):
        start = starts[i]
        curr_chain_id = arr_chain_id[start]
        if curr_chain_id != last_chain_id:
            # New chain
            chain_names[j] = last_chain_id
            res_per_chain[j] = res_counter
            last_chain_id = curr_chain_id
            # Reset residue-per-chain counter
            res_counter = 1
            j += 1
        else:
            res_counter += 1
    # Add last element
    chain_names[j] = last_chain_id
    res_per_chain[j] = res_counter
    j += 1
    # Trim to correct size
    chain_names = chain_names[:j]
    res_per_chain = res_per_chain[:j]
    # Residue IDs from residue starts
    cdef np.ndarray res_ids = arr_res_id[starts[:-1]].astype(np.int32)

    ### Adapt arrays for multiple models
    cdef int model_count = 1
    cdef int chains_per_model = len(chain_names)
    if isinstance(array, AtomArrayStack):
        # Multi-model
        model_count = array.stack_depth()
        chain_names = np.tile(chain_names, model_count)
        res_per_chain = np.tile(res_per_chain, model_count)
        res_ids = np.tile(res_ids, model_count)
        res_types = np.tile(res_types, model_count)
    
    ### Put arrays into file ###
    cdef np.ndarray coord
    if isinstance(array, AtomArrayStack):
        coord = array.coord.reshape(
            (array.stack_depth() * array.array_length(), 3)
        ).astype(np.float32, copy=False)
    else:
        coord = array.coord.astype(np.float32, copy=False)
    file.set_array("xCoordList", coord[:,0], codec=10, param=1000)
    file.set_array("yCoordList", coord[:,1], codec=10, param=1000)
    file.set_array("zCoordList", coord[:,2], codec=10, param=1000)

    file["numModels"] = model_count
    file["chainsPerModel"] = [chains_per_model] * model_count
    file["numChains"] = len(chain_names)
    file.set_array("chainNameList", chain_names, codec=5, param=4)
    file.set_array("chainIdList", chain_names, codec=5, param=4)
    file["groupsPerChain"] = res_per_chain.tolist()
    file["numGroups"] = len(res_ids)
    file.set_array("groupIdList", res_ids, codec=8)
     # Sequence index starts at 0, res IDs at 1
     # -> decrement (the hetero residues (-1) exclusive)
    res_ids[res_ids != -1] -= 1
    file.set_array("sequenceIndexList", res_ids, codec=8)
    file.set_array("groupTypeList", res_types, codec=4)
    file["groupList"] = residues
    file["numAtoms"] = model_count * array_length
