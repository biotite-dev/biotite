# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Patrick Kunzmann"
__all__ = ["get_sequence", "get_model_count", "get_structure", "set_structure",
           "list_assemblies", "get_assembly"]

import itertools
import numpy as np
from ...error import BadStructureError
from ....file import InvalidFileError
from ...atoms import Atom, AtomArray, AtomArrayStack, repeat
from ...filter import filter_first_altloc, filter_highest_occupancy_altloc
from ...box import unitcell_from_vectors, vectors_from_unitcell
from ...util import matrix_rotate
from ....sequence.seqtypes import ProteinSequence, NucleotideSequence
from collections import OrderedDict

_proteinseq_type_list = ["polypeptide(D)", "polypeptide(L)"]
_nucleotideseq_type_list = ["polydeoxyribonucleotide", "polyribonucleotide",
                        "polydeoxyribonucleotide/polyribonucleotide hybrid"
                        ]
_other_type_list = ["cyclic-pseudo-peptide", "other", "peptide nucleic acid",
                        "polysaccharide(D)", "polysaccharide(L)"]

def get_sequence(pdbx_file, data_block=None):
    """
    Get the protein and nucleotide sequences from the
    ``entity_poly.pdbx_seq_one_letter_code_can`` entry.

    Supported polymer types (``_entity_poly.type``) are:
    ``'polypeptide(D)'``, ``'polypeptide(L)'``,
    ``'polydeoxyribonucleotide'``, ``'polyribonucleotide'`` and
    ``'polydeoxyribonucleotide/polyribonucleotide hybrid'``.
    Uracil is converted to Thymine.
    
    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    data_block : string, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
        
    Returns
    -------
    sequences : list of Sequence
        The protein and nucleotide sequences for each entity
        (equivalent to chains in most cases).
    """
    poly_dict = pdbx_file.get_category("entity_poly", data_block)
    seq_string = poly_dict["pdbx_seq_one_letter_code_can"]
    seq_type = poly_dict["type"]
    sequences = []
    if isinstance(seq_string, np.ndarray):
        for string, stype in zip(seq_string, seq_type):
            sequence = _convert_string_to_sequence(string, stype)
            if sequence is not None:
                sequences.append(sequence) 
    else:
        sequences.append(_convert_string_to_sequence(seq_string, seq_type))
    return sequences


def get_model_count(file, data_block=None):
    """
    Get the number of models contained in a :class:`PDBxFile`.

    Parameters
    ----------
    file : PDBxFile
        The file object.
    data_block : str, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.

    Returns
    -------
    model_count : int
        The number of models.
    """
    atom_site_dict = file.get_category("atom_site", data_block)
    return int(atom_site_dict["pdbx_PDB_model_num"][-1])


def get_structure(pdbx_file, model=None, data_block=None, altloc="first",
                  extra_fields=None, use_author_fields=True):
    """
    Create an :class:`AtomArray` or :class:`AtomArrayStack` from the
    ``atom_site`` category in a :class:`PDBxFile`.
    
    Parameters
    ----------
    pdbx_file : PDBxFile
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
    data_block : str, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
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
        The strings in the list are entry names, that are
        additionally added as annotation arrays.
        The annotation category name will be the same as the PDBx
        subcategory name.
        The array type is always `str`.
        An exception are the special field identifiers:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
        These will convert the fitting subcategory into an
        annotation array with reasonable type.
    use_author_fields : bool, optional
        Some fields can be read from two alternative sources,
        for example both, ``label_seq_id`` and ``auth_seq_id`` describe
        the ID of the residue.
        While, the ``label_xxx`` fields can be used as official pointers
        to other categories in the :class:`PDBxFile`, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
        
    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.
        
    Examples
    --------

    >>> import os.path
    >>> file = PDBxFile.read(os.path.join(path_to_structures, "1l2y.cif"))
    >>> arr = get_structure(file, model=1)
    >>> print(len(arr))
    304
    
    """
    altloc = [] if altloc is None else altloc
    extra_fields = [] if extra_fields is None else extra_fields
    
    atom_site_dict = pdbx_file.get_category("atom_site", data_block)
    models = atom_site_dict["pdbx_PDB_model_num"]
    model_count = int(models[-1])
    
    if model is None:
        # For a stack, the annotation are derived from the first model
        model_dict = _get_model_dict(atom_site_dict, 1)
        model_length = len(model_dict["group_PDB"])
        stack = AtomArrayStack(model_count, model_length)
        
        _fill_annotations(stack, model_dict, extra_fields, use_author_fields)
        
        # Check if each model has the same amount of atoms
        # If not, raise exception
        atom_count = len(models)
        if model_length * model_count != atom_count:
            raise InvalidFileError("The models in the file have unequal "
                                   "amount of atoms, give an explicit model "
                                   "instead")
        
        stack.coord = np.zeros((model_count,model_length,3), dtype=np.float32)
        stack.coord[:,:,0] = atom_site_dict["Cartn_x"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,1] = atom_site_dict["Cartn_y"].reshape((model_count,
                                                                model_length))
        stack.coord[:,:,2] = atom_site_dict["Cartn_z"].reshape((model_count,
                                                                model_length))
        
        stack = _filter_altloc(stack, model_dict, altloc)
        
        box = _get_box(pdbx_file, data_block)
        if box is not None:
            # Duplicate same box for each model
            stack.box = np.repeat(box[np.newaxis, ...], model_count, axis=0)
        
        return stack
    
    else:
        if model == 0:
            raise ValueError("The model index must not be 0")
        # Negative models mean model indexing starting from last model
        model = model_count + model + 1 if model < 0 else model
        if model > model_count:
            raise ValueError(
                f"The file has {model_count} models, "
                f"the given model {model} does not exist"
            )

        model_dict = _get_model_dict(atom_site_dict, model)
        model_length = len(model_dict["group_PDB"])
        array = AtomArray(model_length)
        
        _fill_annotations(array, model_dict, extra_fields, use_author_fields)
        
        model_filter = (models == str(model))
        array.coord = np.zeros((model_length, 3), dtype=np.float32)
        array.coord[:,0] = atom_site_dict["Cartn_x"][model_filter] \
                           .astype(np.float32)
        array.coord[:,1] = atom_site_dict["Cartn_y"][model_filter] \
                           .astype(np.float32)
        array.coord[:,2] = atom_site_dict["Cartn_z"][model_filter] \
                           .astype(np.float32)
        
        array = _filter_altloc(array, model_dict, altloc)
        
        array.box = _get_box(pdbx_file, data_block)
        
        return array
        

def _fill_annotations(array, model_dict, extra_fields, use_author_fields):
    prefix = "auth" if use_author_fields else "label"
    array.set_annotation(
        "chain_id", model_dict[f"{prefix}_asym_id"].astype("U4")
    )
    array.set_annotation(
        "res_id", np.array(
            [-1 if e in [".","?"] else int(e)
             for e in model_dict[f"{prefix}_seq_id"]]
        )
    )
    array.set_annotation(
        "ins_code", np.array(
            ["" if e in [".","?"] else e
             for e in model_dict["pdbx_PDB_ins_code"].astype("U1")]
        )
    )
    array.set_annotation(
        "res_name", model_dict[f"{prefix}_comp_id"].astype("U3")
    )
    array.set_annotation(
        "hetero", (model_dict["group_PDB"] == "HETATM")
    )
    array.set_annotation(
        "atom_name", model_dict[f"{prefix}_atom_id"].astype("U6")
    )
    array.set_annotation("element", model_dict["type_symbol"].astype("U2"))
    
    for field in extra_fields:
        if field == "atom_id":
            array.set_annotation(
                "atom_id", model_dict["id"].astype(int)
            )
        elif field == "b_factor":
            array.set_annotation(
                "b_factor", model_dict["B_iso_or_equiv"].astype(float)
            )
        elif field == "occupancy":
            array.set_annotation(
                "occupancy", model_dict["occupancy"].astype(float)
            )
        elif field == "charge":
            array.set_annotation(
                "charge", np.array(
                    [0 if charge in ["?","."] else int(charge)
                     for charge in model_dict["pdbx_formal_charge"]],
                    dtype=int
                )
            )
        else:
            array.set_annotation(field, model_dict[field].astype(str))


def _filter_altloc(array, model_dict, altloc):
    altloc_ids = model_dict.get("label_alt_id")
    occupancy =  model_dict.get("occupancy")
    
    # Filter altloc IDs and return
    if altloc_ids is None:
        return array
    elif altloc == "occupancy" and occupancy is not None:
        return array[
            ...,
            filter_highest_occupancy_altloc(
                array, altloc_ids, occupancy.astype(float)
            )
        ]
    # 'first' is also fallback if file has no occupancy information
    elif altloc == "first":
        return array[..., filter_first_altloc(array, altloc_ids)]
    elif altloc == "all":
        array.set_annotation("altloc_id", altloc_ids)
        return array
    else:
        raise ValueError(f"'{altloc}' is not a valid 'altloc' option")

def _get_model_dict(atom_site_dict, model):
    model_dict = {}
    models = atom_site_dict["pdbx_PDB_model_num"]
    for key in atom_site_dict.keys():
        model_dict[key] = atom_site_dict[key][models == str(model)]
    return model_dict


def _get_box(pdbx_file, data_block):
    if data_block is None:
        cell_dict = pdbx_file.get("cell")
    else:
        cell_dict = pdbx_file.get((data_block, "cell"))
    if cell_dict is None:
        return None
    try:
        len_a, len_b, len_c = [float(cell_dict[length]) for length
                               in ["length_a", "length_b", "length_c"]]
    except ValueError:
        # 'cell_dict' has no proper unit cell values, e.g. '?'
        return None
    alpha, beta, gamma =  [np.deg2rad(float(cell_dict[angle])) for angle
                           in ["angle_alpha", "angle_beta", "angle_gamma"]]
    return vectors_from_unitcell(len_a, len_b, len_c, alpha, beta, gamma)



def set_structure(pdbx_file, array, data_block=None):
    """
    Set the `atom_site` category with an
    :class:`AtomArray` or :class:`AtomArrayStack`.
    
    This will save the coordinates, the mandatory annotation categories
    and the optional annotation categories
    ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
    If the atom array (stack) contains the annotation ``'atom_id'``,
    these values will be used for atom numbering instead of continuous
    numbering.
    
    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    array : AtomArray or AtomArrayStack
        The structure to be written. If a stack is given, each array in
        the stack will be in a separate model.
    data_block : str, optional
        The name of the data block. Default is the first
        (and most times only) data block of the file.
    
    Examples
    --------

    >>> import os.path
    >>> file = PDBxFile()
    >>> set_structure(file, atom_array, data_block="structure")
    >>> file.write(os.path.join(path_to_directory, "structure.cif"))
    
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
    atom_site_dict["label_seq_id"] = np.array(
        ["." if e == -1 else str(e) for e in array.res_id]
    )
    atom_site_dict["pdbx_PDB_ins_code"] = array.ins_code
    atom_site_dict["auth_seq_id"] = atom_site_dict["label_seq_id"]
    atom_site_dict["auth_comp_id"] = atom_site_dict["label_comp_id"]
    atom_site_dict["auth_asym_id"] = atom_site_dict["label_asym_id"]
    atom_site_dict["auth_atom_id"] = atom_site_dict["label_atom_id"]
    
    if "atom_id" in annot_categories:
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
    
    # Write box into file
    if array.box is not None:
        # PDBx files can only store one box for all models
        # -> Use first box
        if array.box.ndim == 3:
            box = array.box[0]
        else:
            box = array.box
        len_a, len_b, len_c, alpha, beta, gamma = unitcell_from_vectors(box)
        cell_dict = OrderedDict()
        cell_dict["length_a"] = "{:6.3f}".format(len_a)
        cell_dict["length_b"] = "{:6.3f}".format(len_b)
        cell_dict["length_c"] = "{:6.3f}".format(len_c)
        cell_dict["angle_alpha"] = "{:5.3f}".format(np.rad2deg(alpha))
        cell_dict["angle_beta"]  = "{:5.3f}".format(np.rad2deg(beta ))
        cell_dict["angle_gamma"] = "{:5.3f}".format(np.rad2deg(gamma))
        pdbx_file.set_category("cell", cell_dict, data_block)


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



def list_assemblies(pdbx_file, data_block=None):
    """
    List the biological assemblies that are available for the structure
    in the given file.

    This function receives the data from the ``pdbx_struct_assembly``
    category in the file.
    Consequently, this category must be present in the file.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    data_block : str, optional
        The name of the data block.
        Defaults to the first (and most times only) data block of the
        file.
    
    Returns
    -------
    assemblies : dict of str -> str
        A dictionary that maps an assembly ID to a description of the
        corresponding assembly.
    """
    assembly_category = pdbx_file.get_category(
        "pdbx_struct_assembly", data_block, expect_looped=True
    )
    if assembly_category is None:
        raise InvalidFileError("File has no 'pdbx_struct_assembly' category")
    return {id: details for id, details in
            zip(assembly_category["id"], assembly_category["details"])}


def get_assembly(pdbx_file, assembly_id=None, model=None, data_block=None,
                 altloc="first", extra_fields=None, use_author_fields=True):
    """
    Build the given biological assembly.

    This function receives the data from the
    ``pdbx_struct_assembly_gen``, ``pdbx_struct_oper_list`` and
    ``atom_site`` categories in the file.
    Consequently, these categories must be present in the file.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The file object.
    assembly_id : str
        The assembly to build
    model : int, optional
        If this parameter is given, the function will return an
        :class:`AtomArray` from the atoms corresponding to the given
        model number (starting at 1).
        Negative values are used to index models starting from the last
        model insted of the first model.
        If this parameter is omitted, an :class:`AtomArrayStack`
        containing all models will be returned, even if the structure
        contains only one model.
    data_block : str, optional
        The name of the data block.
        Defaults to the first (and most times only) data block of the
        file.
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
        The strings in the list are entry names, that are
        additionally added as annotation arrays.
        The annotation category name will be the same as the PDBx
        subcategory name.
        The array type is always `str`.
        An exception are the special field identifiers:
        ``'atom_id'``, ``'b_factor'``, ``'occupancy'`` and ``'charge'``.
        These will convert the fitting subcategory into an
        annotation array with reasonable type.
    use_author_fields : bool, optional
        Some fields can be read from two alternative sources,
        for example both, ``label_seq_id`` and ``auth_seq_id`` describe
        the ID of the residue.
        While, the ``label_xxx`` fields can be used as official pointers
        to other categories in the :class:`PDBxFile`, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
    
    Returns
    -------
    assembly : AtomArray or AtomArrayStack
        The assembly. The return type depends on the `model` parameter.
    """
    assembly_gen_category = pdbx_file.get_category(
        "pdbx_struct_assembly_gen", data_block, expect_looped=True
    )
    if assembly_gen_category is None:
        raise InvalidFileError(
            "File has no 'pdbx_struct_assembly_gen' category"
        )

    struct_oper_category = pdbx_file.get_category(
        "pdbx_struct_oper_list", data_block, expect_looped=True
    )
    if struct_oper_category is None:
        raise InvalidFileError("File has no 'pdbx_struct_oper_list' category")

    if assembly_id is None:
        assembly_id = assembly_gen_category["assembly_id"][0]
    elif assembly_id not in assembly_gen_category["assembly_id"]:
        raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    transformations = _get_transformations(struct_oper_category)


    ### Get transformations to apply and the affected asym IDs
    # Find the operation expression for given assembly ID
    # We already asserted that the ID is actually present
    for id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"],
        assembly_gen_category["oper_expression"],
        assembly_gen_category["asym_id_list"]
    ):
        if id == assembly_id:
            operations = _parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")
            break
    
    
    ### Get structure containing the affected asym IDs
    # Include 'label_asym_id' as annotation array
    # for correct asym ID filtering
    extra_fields = [] if extra_fields is None else extra_fields
    if "label_asym_id" in extra_fields:
        extra_fields_and_asym = extra_fields
    else:
        extra_fields_and_asym = extra_fields + ["label_asym_id"]
    structure = get_structure(
        pdbx_file, model, data_block,
        altloc, extra_fields_and_asym, use_author_fields
    )
    # Filter asym IDs
    structure = structure[..., np.isin(structure.label_asym_id, asym_ids)]
    # Remove 'label_asym_id', if it was not included in the original
    # user-supplied 'extra_fields'
    if "label_asym_id" not in extra_fields:
        structure.del_annotation("label_asym_id")


    ### Prepare coordinates
    if model is None:
        # Coordinates for AtomArrayStack
        assembly_coord = np.zeros(
            (len(operations), structure.stack_depth(),
             structure.array_length(), 3)
        )
    else:
        # Coordinates for AtomArray
        assembly_coord = np.zeros(
            (len(operations), structure.array_length(), 3)
        )
    

    ### Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = structure.coord
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            rotation_matrix, translation_vector = transformations[op_step]
            # Rotate
            coord = matrix_rotate(coord, rotation_matrix)
            # Translate
            coord += translation_vector
        assembly_coord[i] = coord
    
    return repeat(structure, assembly_coord)     


def _get_transformations(struct_oper):
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"]):
        rotation_matrix = np.array([
            [float(struct_oper[f"matrix[{i}][{j}]"][index]) for j in (1,2,3)]
            for i in (1,2,3)
        ])
        translation_vector = np.array([
            float(struct_oper[f"vector[{i}]"][index]) for i in (1,2,3)
        ])
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict


def _parse_operation_expression(expression):
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis 
    expressions_per_step = expression.replace(")","").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for expr in expressions_per_step:
        if "-" in expr:
            # Range of operation IDs, they must be integers
            first, last = expr.split("-")
            operations.append(
                [str(id) for id in range(int(first), int(last)+1)]
            )
        elif "," in expr:
            # List of operation IDs
            operations.append(expr.split(","))
        else:
            # Single operation ID
            operations.append([expr])

    # Cartesian product of operations
    return list(itertools.product(*operations))


def _convert_string_to_sequence(string, stype):
    # Convert strings to ProteinSequence-Object if stype is
    # contained in _proteinseq_type_list or to NucleotideSequence-
    # Object if stype is contained in _nucleotideseq_type_list
    if stype in _proteinseq_type_list:
        return ProteinSequence(string)
    elif stype in _nucleotideseq_type_list:
        string = string.replace("U", "T")
        return NucleotideSequence(string)
    elif stype in _other_type_list:
        return None
    else:
        raise InvalidFileError("mmCIF _entity_poly.type unsupported"
                               " type: " + stype)
