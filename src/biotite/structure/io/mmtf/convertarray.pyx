# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["set_structure"]

cimport cython
cimport numpy as np

import numpy as np
from .file import MMTFFile
from ...atoms import Atom, AtomArray, AtomArrayStack
from ...bonds import BondList
from ...error import BadStructureError
from ...residues import get_residue_starts
from ...box import unitcell_from_vectors
from ...info.misc import link_type

ctypedef np.int8_t int8
ctypedef np.int16_t int16
ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32


def set_structure(file, array):
    """
    set_structure(file, array)

    Set the relevant fields of an MMTF file with the content of an
    :class:`AtomArray` or :class:`AtomArrayStack`.
    
    All required and some optional fields of the MMTF file will be set
    or overriden if the field does already exist. Fields are removed
    when they are optional and when setting the structure information
    could invalidate its content (e.g. altLocList). 
    
    Parameters
    ----------
    file : MMTFFile
        The file object.
    array : AtomArray or AtomArrayStack
        The structure to be written. If a stack is given, each array in
        the stack will be in a separate model.
    
    Notes
    -----
    As the MMTF format only supports one unit cell, individual unit
    cells for each model are not supported.
    Instead only the first box in an :class:`AtomArrayStack` is written
    into the file.
    
    Examples
    --------

    >>> import os.path
    >>> file = MMTFFile()
    >>> set_structure(file, atom_array)
    >>> file.write(os.path.join(path_to_directory, "structure.mmtf"))
    
    """
    cdef bint include_bonds = (array.bonds is not None)
    
    cdef int i=0, j=0
    cdef array_length = array.array_length()
    

    # Get annotation arrays from atom array (stack)
    cdef np.ndarray arr_chain_id  = array.chain_id
    cdef np.ndarray arr_res_id    = array.res_id
    cdef np.ndarray arr_ins_code  = array.ins_code
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
    # List of 'groupType' dictsfor setting the file's 'groupList'
    cdef list residues
    # Maps 'groupType' values (not the keys) to the index in 'residues'
    # Necessary a 'groupType' are dictionaries, which are not hashable
    cdef dict residue_dict
    # An entry in 'residues'
    cdef dict group_type
    # An entry in 'residue_dict'
    cdef tuple hashable_group_type
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
    res_types = np.zeros(len(starts)-1, dtype=np.int32)
    residues = []
    residue_dict = {}
    for i in range(len(starts)-1):
        start = starts[i]
        stop = starts[i+1]
        res_length = stop - start
        res_name = arr_res_name[start]
        # Get intra-residue bonds of this residue
        if include_bonds:
            intra_bonds = array.bonds[start:stop]
        
        # Create 'groupType' dictionary for current residue
        group_type = {}
        group_type["atomNameList"] = tuple(
            arr_atom_name[start:stop].tolist()
        )
        group_type["elementList"] = tuple(
            [e.capitalize() for e in arr_element[start:stop]]
        )
        if arr_charge is not None:
            group_type["formalChargeList"] = tuple(
                arr_charge[start:stop].tolist()
            )
        else:
            group_type["formalChargeList"] = (0,) * (stop-start)
        group_type["groupName"] = res_name
        link = link_type(res_name)
        # Use 'NON-POLYMER' as default
        if link is None:
            link = "NON-POLYMER"
        group_type["chemCompType"] = link
        # Add intra-residue bonds
        if include_bonds:
            intra_bonds = array.bonds[start:stop]
            bond_array = intra_bonds.as_array()
            group_type["bondAtomList"] = tuple(
                bond_array[:,:2].flatten().tolist()
            )
            group_type["bondOrderList"] = tuple(
                bond_array[:,2].tolist()
            )
        else:
            group_type["bondAtomList"] = ()
            group_type["bondOrderList"] = ()
        
        # Find index of current residue in later 'groupList'
        hashable_group_type = tuple(group_type.values())
        residue_i = residue_dict.get(hashable_group_type, -1)
        if residue_i == -1:
            # Add new residue if not yet existing in 'groupList'
            residue_i = len(residues)
            residues.append(group_type)
            residue_dict[hashable_group_type] = residue_i

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
    cdef np.ndarray res_inscodes
    res_inscodes = arr_ins_code[starts[:-1]]

    ### Adapt arrays for multiple models
    cdef int model_count = 1
    cdef int chains_per_model = len(chain_names)
    if isinstance(array, AtomArrayStack):
        # Multi-model
        model_count = array.stack_depth()
        chain_names = np.tile(chain_names, model_count)
        res_per_chain = np.tile(res_per_chain, model_count)
        res_ids = np.tile(res_ids, model_count)
        res_inscodes = np.tile(res_inscodes, model_count)
        res_types = np.tile(res_types, model_count)


    ### Remove arrays from file ###
    # Arrays are removed if they are optional
    # and if setting the structure information invalidates its content
    _delete_record(file, "bondAtomList")
    _delete_record(file, "bondOrderList")
    _delete_record(file, "bFactorList")
    _delete_record(file, "atomIdList")
    _delete_record(file, "altLocList")
    _delete_record(file, "occupancyList")
    _delete_record(file, "secStructList")
    _delete_record(file, "insCodeList")


    ### Put prepared arrays into file ###
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
    file.set_array("insCodeList", res_inscodes, codec=6)
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
    

    ### Add unit cell ###
    if array.box is not None:
        if isinstance(array, AtomArray):
            box = array.box
        elif isinstance(array, AtomArrayStack):
            # Use box of first model, since MMTF does not support
            # multiple boxes
            box = array.box[0]
        len_a, len_b, len_c, alpha, beta, gamma = unitcell_from_vectors(box)
        file["unitCell"] = [
            len_a, len_b, len_c,
            np.rad2deg(alpha), np.rad2deg(beta), np.rad2deg(gamma)
        ]
    
    
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