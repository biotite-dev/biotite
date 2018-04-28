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

__all__ = ["set_structure"]


def set_structure(file, array):
    cdef bint include_bonds = (array.bonds is not None)
    
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
    # BondList for inter-residue bonds
    # intra-residue bonds are successively removed
    if include_bonds:
        inter_bonds = array.bonds.copy()
    # 'len(starts)-1' since 'starts' has the end
    # of the atom array appended
    residue_i_array = np.zeros(len(starts)-1, dtype=np.int32)
    res_types = np.zeros(len(starts)-1, dtype=np.int32)
    residues = []
    # Dictionary maps (name, length) tuples to indices in residue list
    # (later groupList)
    res_tuple_dict = {}
    for i in range(len(starts)-1):
        start = starts[i]
        stop = starts[i+1]
        res_length = stop - start
        res_name = arr_res_name[start]
        # Get intra-residue bonds of this residue
        if include_bonds:
            intra_bonds = array.bonds[start:stop]
        # Check if the residue does already exist
        # (same name and length)
        res_tuple = (res_name, res_length)
        residue_i = res_tuple_dict.get(res_tuple, -1)
        if residue_i == -1:
            # If it does not exist, create a new entry in dictionary
            curr_res = {}
            curr_res["atomNameList"] = arr_atom_name[start:stop].tolist()
            curr_res["elementList"] = [e.capitalize() for e
                                       in arr_element[start:stop]]
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
            # Add intra-residue bonds
            if include_bonds:
                intra_bonds = array.bonds[start:stop]
                bond_array = intra_bonds.as_array()
                curr_res["bondAtomList"] = bond_array[:,:2].flatten().tolist()
                curr_res["bondOrderList"] = bond_array[:,2].tolist()
            else:
                curr_res["bondAtomList"] = []
                curr_res["bondOrderList"] = []
            # Add new residue to list
            residue_i = len(residues)
            residues.append(curr_res)
            res_tuple_dict[res_tuple] = residue_i
        # Remove intra-residue bonds from all bonds
        # to obtain inter-residue bonds
        # If the residue is already known is irrelevant for this case
        if include_bonds:
            # Offset is required to obtain original indices
            # for bond removal
            intra_bonds.offset_indices(start)
            inter_bonds.remove_bonds(intra_bonds)
        # Put new or already known residue to sequence of residue types
        res_types[i] = residue_i
    
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

    ### Remove arrays from file ###
    # Arrays are removed when they are optional
    # and when setting the structure information invalidates its content
    _delete_record(file, "bondAtomList")
    _delete_record(file, "bondOrderList")
    _delete_record(file, "bFactorList")
    _delete_record(file, "atomIdList")
    _delete_record(file, "altLocList")
    _delete_record(file, "occupancyList")
    _delete_record(file, "secStructList")
    _delete_record(file, "insCodeList")

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
    # Optional annotation arrays
    categories = array.get_annotation_categories()
    if "atom_id" in categories:
        file.set_array("atomIdList",
                       np.tile(array.atom_id.astype(np.int32), model_count),
                       codec=8)
    if "b_factor" in categories:
        file.set_array("bFactorList",
                       np.tile(array.b_factor.astype(np.float32), model_count),
                       codec=10, param=100)
    if "occupancy" in categories:
        file.set_array("occupancyList",
                       np.tile(array.occupancy.astype(np.float32), model_count),
                       codec=9, param=100)

    ### Add inter-residue bonds ###
    if include_bonds:
        all_inter_bonds = inter_bonds
        # Repeat the inter-residue bonds for each additional model
        for i in range(model_count-1):
            all_inter_bonds += inter_bonds
        bond_array = all_inter_bonds.as_array()
        file.set_array("bondAtomList",
                       bond_array[:,:2].flatten().astype(np.int32),
                       codec=4)
        file.set_array("bondOrderList",
                       bond_array[:,2].astype(np.int8),
                       codec=2)
        file["numBonds"] = array.bonds.get_bond_count() * model_count
    else:
        file["numBonds"] = 0
    
    ### Add additional information ###
    # Only set additional information, if not already set
    try:
        val = file["mmtfVersion"]
    except KeyError:
        file["mmtfVersion"] = "1.0.0"
    try:
        val = file["mmtfProducer"]
    except KeyError:
        file["mmtfProducer"] = "UNKNOWN"


def _delete_record(file, record):
    try:
        del file[record]
    except:
        pass