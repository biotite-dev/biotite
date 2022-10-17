# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["get_model_count", "get_structure"]

cimport cython
cimport numpy as np

import numpy as np
from .file import MMTFFile
from ...atoms import Atom, AtomArray, AtomArrayStack
from ...bonds import BondList
from ...error import BadStructureError
from ...filter import filter_first_altloc, filter_highest_occupancy_altloc
from ...residues import get_residue_starts
from ...box import vectors_from_unitcell
from ....file import InvalidFileError

ctypedef np.int8_t int8
ctypedef np.int16_t int16
ctypedef np.int32_t int32
ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.uint32_t uint32
ctypedef np.uint64_t uint64
ctypedef np.float32_t float32


def get_model_count(file):
    """
    Get the number of models contained in a MMTF file.

    Parameters
    ----------
    file : MMTFFile
        The file object.

    Returns
    -------
    model_count : int
        The number of models.
    """
    return file["numModels"]

    
def get_structure(file, model=None, altloc="first",
                  extra_fields=[], include_bonds=False):
    """
    get_structure(file, model=None, altloc=[], extra_fields=[],
                  include_bonds=False)
    
    Get an :class:`AtomArray` or :class:`AtomArrayStack` from the MMTF file.
    
    Parameters
    ----------
    file : MMTFFile
        The file object.
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the last
        model insted of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the structure
        contains only one model.
    altloc : {'first', 'occupancy', 'all'}
        This parameter defines how *altloc* IDs are handled:
            - ``'first'`` - Use atoms that have the first *altloc* ID
              appearing in a residue.
            - ``'occupancy'`` - Use atoms that have the *altloc* ID
              with the highest occupancy for a residue.
            - ``'all'`` - Use all atoms.
              Note that this leads to duplicate atoms.
              When this option is chosen, the ``altloc_id`` annotation
              array is added to the returned structure.
    extra_fields : list of str, optional
        The strings in the list are optional annotation categories
        that should be stored in the output array or stack.
        These are valid values:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
    
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
    
    Examples
    --------

    >>> import os.path
    >>> file = MMTFFile.read(os.path.join(path_to_structures, "1l2y.mmtf"))
    >>> array = get_structure(file, model=1)
    >>> print(array.array_length())
    304
    >>> stack = get_structure(file)
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
    cdef np.ndarray index_list = file["groupIdList"]
    cdef int32[:] res_ids = index_list
    cdef np.ndarray x_coord = file["xCoordList"]
    cdef np.ndarray y_coord = file["yCoordList"]
    cdef np.ndarray z_coord = file["zCoordList"]
    cdef np.ndarray occupancy = file.get("occupancyList")
    cdef np.ndarray b_factor
    if "b_factor" in extra_fields:
        b_factor = file["bFactorList"]
    cdef np.ndarray atom_ids
    if "atom_id" in extra_fields:
        atom_ids = file["atomIdList"]
    cdef np.ndarray all_altloc_ids
    cdef np.ndarray inscode
    all_altloc_ids = file.get("altLocList")
    inscode = file.get("insCodeList")
    

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
    cdef np.ndarray hetero_res = np.zeros(len(group_list), dtype=bool)
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
    cdef np.ndarray altloc_ids
    cdef np.ndarray inscode_array
    
    
    if model == None:
        lengths = _get_model_lengths(res_type_i, chains_per_model,
                                     res_per_chain, atoms_per_res)
        # Check if each model has the same amount of atoms
        # If not, raise exception
        if (lengths != lengths[0]).any():
            raise InvalidFileError("The models in the file have unequal "
                                   "amount of atoms, give an explicit "
                                   "model instead")
        length = lengths[0]

        depth = model_count
        
        
        array = AtomArrayStack(depth, length)
        array.coord = np.stack(
            [x_coord,
             y_coord,
             z_coord],
             axis=1
        ).reshape(depth, length, 3)
        
        # Create altloc array for the final filtering
        if all_altloc_ids is not None:
            altloc_ids = all_altloc_ids[:length]
        else:
            altloc_ids = None
        
        extra_charge = False
        if "ins_code" in extra_fields:
            extra_inscode = True
            array.add_annotation("ins_code", "U1")
        if "charge" in extra_fields:
            extra_charge = True
            array.add_annotation("charge", int)
        if "atom_id" in extra_fields:
            array.set_annotation("atom_id", atom_ids[:length])
        if "b_factor" in extra_fields:
            array.set_annotation("b_factor", b_factor[:length])
        if "occupancy" in extra_fields:
            array.set_annotation("occupancy", occupancy[:length])
        
        _fill_annotations(1, array, extra_charge,
                          chain_names, chains_per_model, res_per_chain,
                          res_type_i, res_ids, inscode, atoms_per_res,
                          res_names, hetero_res, atom_names, elements, charges)
        
        if include_bonds:
            array.bonds = _create_bond_list(
                1, file["bondAtomList"], file["bondOrderList"],
                0, length, file["numAtoms"], group_list, res_type_i,
                atoms_per_res, res_per_chain, chains_per_model
            )
    

    else:
        lengths = _get_model_lengths(res_type_i, chains_per_model,
                                     res_per_chain, atoms_per_res)
        if model == 0:
            raise ValueError("The model index must not be 0")
        # Negative models mean model index starting from last model
        model = len(lengths) + model + 1 if model < 0 else model
        if model > len(lengths):
            raise ValueError(
                f"The file has {len(lengths)} models, "
                f"the given model {model} does not exist"
            )

        length = lengths[model-1]
        # Indices to filter coords and some annotations
        # for the specified model
        start_i = np.sum(lengths[:model-1])
        stop_i = start_i + length
        
        array = AtomArray(length)
        array.coord[:,0] = x_coord[start_i : stop_i]
        array.coord[:,1] = y_coord[start_i : stop_i]
        array.coord[:,2] = z_coord[start_i : stop_i]
        
        # Create altloc array for the final filtering
        if all_altloc_ids is not None:
            altloc_ids = np.array(all_altloc_ids[start_i : stop_i], dtype="U1")
        else:
            altloc_ids = None
        
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
        
        _fill_annotations(model, array, extra_charge,
                          chain_names, chains_per_model, res_per_chain,
                          res_type_i, res_ids, inscode, atoms_per_res,
                          res_names, hetero_res, atom_names, elements, charges)
        
        if include_bonds:
            array.bonds = _create_bond_list(
                model, file["bondAtomList"], file["bondOrderList"],
                start_i, stop_i, file["numAtoms"], group_list, res_type_i,
                atoms_per_res, res_per_chain, chains_per_model
            )
    
    # Get box
    if "unitCell" in file:
        a_len, b_len, c_len, alpha, beta, gamma = file["unitCell"]
        alpha = np.deg2rad(alpha)
        beta  = np.deg2rad(beta )
        gamma = np.deg2rad(gamma)
        box = vectors_from_unitcell(
            a_len, b_len, c_len, alpha, beta, gamma
        )
        if isinstance(array, AtomArrayStack):
            array.box = np.repeat(
                box[np.newaxis, ...], array.stack_depth(), axis=0
            )
        else:
            # AtomArray
            array.box = box
    
    
    # Filter altloc IDs and return
    if altloc_ids is None:
        return array
    elif altloc == "occupancy" and occupancy is not None:
        return array[
            ...,
            filter_highest_occupancy_altloc(array, altloc_ids, occupancy)
        ]
    # 'first' is also fallback if file has no occupancy information
    elif altloc == "first":
        return array[..., filter_first_altloc(array, altloc_ids)]
    elif altloc == "all":
        array.set_annotation("altloc_id", altloc_ids)
        return array
    else:
        raise ValueError(f"'{altloc}' is not a valid 'altloc' option")


def _get_model_lengths(int32[:] res_type_i,
                       int32[:] chains_per_model,
                       int32[:] res_per_chain,
                       int32[:] atoms_per_res):
    cdef int[:] model_lengths = np.zeros(len(chains_per_model), np.int32)
    cdef int atom_count = 0
    cdef int model_i = 0
    cdef int chain_i = 0
    cdef int res_i
    cdef int res_count_in_chain = 0
    cdef int chain_count_in_model = 0
    # The length of 'res_type_i'
    # is equal to the total number of residues
    for res_i in range(res_type_i.shape[0]):
        atom_count += atoms_per_res[res_type_i[res_i]]
        res_count_in_chain += 1
        if res_count_in_chain == res_per_chain[chain_i]:
            # Chain is full -> Bump chain index and reset residue count
            res_count_in_chain = 0
            chain_i += 1
            chain_count_in_model += 1
        if chain_count_in_model == chains_per_model[model_i]:
            # Model is full -> Bump model index and reset chain count
            chain_count_in_model = 0
            model_lengths[model_i] = atom_count
            # Restart counting for the next model
            atom_count = 0
            model_i += 1
    return np.asarray(model_lengths)

    
def _fill_annotations(int model, array,
                      bint extra_charge,
                      np.ndarray chain_names,
                      int32[:] chains_per_model,
                      int32[:] res_per_chain,
                      int32[:] res_type_i,
                      int32[:] res_ids,
                      np.ndarray res_inscodes,
                      np.ndarray atoms_per_res,
                      np.ndarray res_names,
                      np.ndarray hetero_res,
                      np.ndarray atom_names,
                      np.ndarray elements,
                      np.ndarray charges):
    # Get annotation arrays from atom array (stack)
    cdef np.ndarray chain_id  = array.chain_id
    cdef np.ndarray res_id    = array.res_id
    cdef np.ndarray ins_code  = array.ins_code
    cdef np.ndarray res_name  = array.res_name
    cdef np.ndarray hetero    = array.hetero
    cdef np.ndarray atom_name = array.atom_name
    cdef np.ndarray element   = array.element
    if extra_charge:
        charge = array.charge

    cdef int model_i = 0
    cdef int chain_i = 0
    cdef int res_i
    cdef int atom_i = 0
    cdef int res_count_in_chain = 0
    cdef int chain_count_in_model = 0
    cdef int atom_index_in_res

    cdef chain_id_for_chain
    cdef res_name_for_res
    cdef inscode_for_res
    cdef bint hetero_for_res
    cdef int res_id_for_res
    cdef int type_i

    # The length of 'res_type_i'
    # is equal to the total number of residues
    for res_i in range(res_type_i.shape[0]):
        # Wait for the data of the given model
        if model_i == model-1: 
            chain_id_for_chain = chain_names[chain_i]
            res_id_for_res = res_ids[res_i]
            if res_inscodes is not None:
                inscode_for_res = res_inscodes[res_i]
            type_i = res_type_i[res_i]
            res_name_for_res = res_names[type_i]
            hetero_for_res = hetero_res[type_i]

            for atom_index_in_res in range(atoms_per_res[type_i]):
                chain_id[atom_i]  = chain_id_for_chain
                res_id[atom_i]    = res_id_for_res
                ins_code[atom_i]  = inscode_for_res
                hetero[atom_i]    = hetero_for_res
                res_name[atom_i]  = res_name_for_res
                atom_name[atom_i] = atom_names[type_i][atom_index_in_res]
                element[atom_i]   = elements[type_i][atom_index_in_res].upper()
                if extra_charge:
                    charge[atom_i] = charges[type_i][atom_index_in_res]
                atom_i += 1
        
        elif model_i > model-1:
            # The given model has already been parsed
            # -> parsing is finished
            break
        
        res_count_in_chain += 1
        if res_count_in_chain == res_per_chain[chain_i]:
            # Chain is full -> Bump chain index and reset residue count
            res_count_in_chain = 0
            chain_i += 1
            chain_count_in_model += 1
        if chain_count_in_model == chains_per_model[model_i]:
            # Model is full -> Bump model index and reset chain count
            chain_count_in_model = 0
            model_i += 1


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
    # Dictionary for groupList entry
    cdef dict residue
    # Fill the array
    for i in range(len(group_list)):
        residue = group_list[i]
        bonds_in_residue = np.array(residue["bondAtomList"], dtype=np.uint32)
        intra_bonds[i, :bonds_per_res[i], :2] = \
            np.array(residue["bondAtomList"], dtype=np.uint32).reshape((-1, 2))
        intra_bonds[i, :bonds_per_res[i], 2] = residue["bondOrderList"]

    # Unify intra-residue bonds to one BondList
    cdef int model_i = 0
    cdef int chain_i = 0
    cdef int res_i
    cdef int res_count_in_chain = 0
    cdef int chain_count_in_model = 0
    cdef int type_i
    intra_bond_list = BondList(0)
    # The length of 'res_type_i'
    # is equal to the total number of residues
    for res_i in range(res_type_i.shape[0]):
        # Wait for the data of the given model
        if model_i == model-1: 
            type_i = res_type_i[res_i]
            bond_list_per_res = BondList(
                atoms_per_res[type_i],
                intra_bonds[type_i, :bonds_per_res[type_i]]
            )
            intra_bond_list += bond_list_per_res
        
        elif model_i > model-1:
            # The given model has already been parsed
            # -> parsing is finished
            break

        res_count_in_chain += 1
        if res_count_in_chain == res_per_chain[chain_i]:
            # Chain is full -> Bump chain index and reset residue count
            res_count_in_chain = 0
            chain_i += 1
            chain_count_in_model += 1
        if chain_count_in_model == chains_per_model[model_i]:
            # Model is full -> Bump model index and reset chain count
            chain_count_in_model = 0
            model_i += 1
    
    # Add inter-residue bonds to BondList
    cdef np.ndarray inter_bonds = np.zeros((len(bond_types), 3),
                                           dtype=np.uint32)
    inter_bonds[:,:2] = bonds.reshape((len(bond_types), 2))
    inter_bonds[:,2] = bond_types
    inter_bond_list = BondList(atom_count, inter_bonds)
    inter_bond_list = inter_bond_list[model_start : model_stop]
    global_bond_list = inter_bond_list.merge(intra_bond_list)
    return global_bond_list