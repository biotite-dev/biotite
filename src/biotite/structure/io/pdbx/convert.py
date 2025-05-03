# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.pdbx"
__author__ = "Fabrice Allain, Patrick Kunzmann, Cheyenne Ziegler"
__all__ = [
    "get_sequence",
    "get_model_count",
    "get_structure",
    "set_structure",
    "get_component",
    "set_component",
    "list_assemblies",
    "get_assembly",
    "get_unit_cell",
    "get_sse",
]

import itertools
import warnings
from collections import defaultdict
import numpy as np
from biotite.file import InvalidFileError
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.structure.atoms import (
    AtomArray,
    AtomArrayStack,
    concatenate,
    repeat,
)
from biotite.structure.bonds import BondList, BondType, connect_via_residue_names
from biotite.structure.box import (
    coord_to_fraction,
    fraction_to_coord,
    space_group_transforms,
    unitcell_from_vectors,
    vectors_from_unitcell,
)
from biotite.structure.error import BadStructureError
from biotite.structure.filter import _canonical_aa_list as canonical_aa_list
from biotite.structure.filter import (
    _canonical_nucleotide_list as canonical_nucleotide_list,
)
from biotite.structure.filter import (
    filter_first_altloc,
    filter_highest_occupancy_altloc,
)
from biotite.structure.geometry import centroid
from biotite.structure.io.pdbx.bcif import (
    BinaryCIFBlock,
    BinaryCIFColumn,
    BinaryCIFFile,
)
from biotite.structure.io.pdbx.cif import CIFBlock, CIFFile
from biotite.structure.io.pdbx.component import MaskValue
from biotite.structure.io.pdbx.encoding import StringArrayEncoding
from biotite.structure.residues import (
    get_residue_count,
    get_residue_positions,
    get_residue_starts_for,
)
from biotite.structure.transform import AffineTransformation

# Bond types in `struct_conn` category that refer to covalent bonds
PDBX_BOND_TYPE_ID_TO_TYPE = {
    # Although a covalent bond, could in theory have a higher bond order,
    # practically inter-residue bonds are always single
    "covale": BondType.SINGLE,
    "covale_base": BondType.SINGLE,
    "covale_phosphate": BondType.SINGLE,
    "covale_sugar": BondType.SINGLE,
    "disulf": BondType.SINGLE,
    "modres": BondType.SINGLE,
    "modres_link": BondType.SINGLE,
    "metalc": BondType.COORDINATION,
}
PDBX_BOND_TYPE_TO_TYPE_ID = {
    BondType.ANY: "covale",
    BondType.SINGLE: "covale",
    BondType.DOUBLE: "covale",
    BondType.TRIPLE: "covale",
    BondType.QUADRUPLE: "covale",
    BondType.AROMATIC_SINGLE: "covale",
    BondType.AROMATIC_DOUBLE: "covale",
    BondType.AROMATIC_TRIPLE: "covale",
    BondType.COORDINATION: "metalc",
}
PDBX_BOND_TYPE_TO_ORDER = {
    BondType.SINGLE: "sing",
    BondType.DOUBLE: "doub",
    BondType.TRIPLE: "trip",
    BondType.QUADRUPLE: "quad",
    BondType.AROMATIC_SINGLE: "sing",
    BondType.AROMATIC_DOUBLE: "doub",
    BondType.AROMATIC_TRIPLE: "trip",
    # These are masked later, it is merely added here to avoid a KeyError
    BondType.ANY: "",
    BondType.AROMATIC: "",
    BondType.COORDINATION: "",
}
# Map 'chem_comp_bond' bond orders and aromaticity to 'BondType'...
COMP_BOND_ORDER_TO_TYPE = {
    ("SING", "N"): BondType.SINGLE,
    ("DOUB", "N"): BondType.DOUBLE,
    ("TRIP", "N"): BondType.TRIPLE,
    ("QUAD", "N"): BondType.QUADRUPLE,
    ("SING", "Y"): BondType.AROMATIC_SINGLE,
    ("DOUB", "Y"): BondType.AROMATIC_DOUBLE,
    ("TRIP", "Y"): BondType.AROMATIC_TRIPLE,
    ("AROM", "Y"): BondType.AROMATIC,
}
# ...and vice versa
COMP_BOND_TYPE_TO_ORDER = {
    bond_type: order for order, bond_type in COMP_BOND_ORDER_TO_TYPE.items()
}
CANONICAL_RESIDUE_LIST = canonical_aa_list + canonical_nucleotide_list
# it was observed that when the number or rows in `atom_site` and `struct_conn`
# exceed a certain threshold,
# a dictionary approach is less computation and memory intensive than the dense
# vectorized approach.
# https://github.com/biotite-dev/biotite/pull/765#issuecomment-2708867357
FIND_MATCHES_SWITCH_THRESHOLD = 4000000

_proteinseq_type_list = ["polypeptide(D)", "polypeptide(L)"]
_nucleotideseq_type_list = [
    "polydeoxyribonucleotide",
    "polyribonucleotide",
    "polydeoxyribonucleotide/polyribonucleotide hybrid",
]
_other_type_list = [
    "cyclic-pseudo-peptide",
    "other",
    "peptide nucleic acid",
    "polysaccharide(D)",
    "polysaccharide(L)",
]


def _filter(category, index):
    """
    Reduce the given category to the values selected by the given index,
    """
    Category = type(category)
    Column = Category.subcomponent_class()
    Data = Column.subcomponent_class()

    return Category(
        {
            key: Column(
                Data(column.data.array[index]),
                (Data(column.mask.array[index]) if column.mask is not None else None),
            )
            for key, column in category.items()
        }
    )


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
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.

    Returns
    -------
    sequence_dict : Dictionary of Sequences
        Dictionary keys are derived from ``entity_poly.pdbx_strand_id``
        (equivalent to ``atom_site.auth_asym_id``).
        Dictionary values are sequences.

    Notes
    -----
    The ``entity_poly.pdbx_seq_one_letter_code_can`` field contains the initial
    complete sequence. If the structure represents a truncated or spliced
    version of this initial sequence, it will include only a subset of the
    initial sequence. Use biotite.structure.get_residues to retrieve only
    the residues that are represented in the structure.
    """

    block = _get_block(pdbx_file, data_block)
    poly_category = block["entity_poly"]

    seq_string = poly_category["pdbx_seq_one_letter_code_can"].as_array(str)
    seq_type = poly_category["type"].as_array(str)

    sequences = [
        _convert_string_to_sequence(string, stype)
        for string, stype in zip(seq_string, seq_type)
    ]

    strand_ids = poly_category["pdbx_strand_id"].as_array(str)
    strand_ids = [strand_id.split(",") for strand_id in strand_ids]

    sequence_dict = {
        strand_id: sequence
        for sequence, strand_ids in zip(sequences, strand_ids)
        for strand_id in strand_ids
        if sequence is not None
    }

    return sequence_dict


def get_model_count(pdbx_file, data_block=None):
    """
    Get the number of models contained in a file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.

    Returns
    -------
    model_count : int
        The number of models.
    """
    block = _get_block(pdbx_file, data_block)
    return len(np.unique((block["atom_site"]["pdbx_PDB_model_num"].as_array(np.int32))))


def get_structure(
    pdbx_file,
    model=None,
    data_block=None,
    altloc="first",
    extra_fields=None,
    use_author_fields=True,
    include_bonds=False,
):
    """
    Create an :class:`AtomArray` or :class:`AtomArrayStack` from the
    ``atom_site`` category in a file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
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
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
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
        to other categories in the file, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
        If the requested field is not available, the respective other
        field is taken as fallback.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Inter-residue bonds, will be read from the ``struct_conn``
        category.
        Intra-residue bonds will be read from the ``chem_comp_bond``, if
        available, otherwise they will be derived from the Chemical
        Component Dictionary.

    Returns
    -------
    array : AtomArray or AtomArrayStack
        The return type depends on the `model` parameter.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1l2y.cif"))
    >>> arr = get_structure(file, model=1)
    >>> print(len(arr))
    304
    """
    block = _get_block(pdbx_file, data_block)

    extra_fields = set() if extra_fields is None else set(extra_fields)

    atom_site = block.get("atom_site")
    if atom_site is None:
        raise InvalidFileError("Missing 'atom_site' category in file")

    models = atom_site["pdbx_PDB_model_num"].as_array(np.int32)
    model_count = len(np.unique(models))
    atom_count = len(models)

    if model is None:
        # For a stack, the annotations are derived from the first model
        model_atom_site = _filter_model(atom_site, 1)
        # Any field of the category would work here to get the length
        model_length = model_atom_site.row_count
        atoms = AtomArrayStack(model_count, model_length)

        # Check if each model has the same amount of atoms
        # If not, raise exception
        if model_length * model_count != atom_count:
            raise InvalidFileError(
                "The models in the file have unequal "
                "amount of atoms, give an explicit model "
                "instead"
            )

        atoms.coord[:, :, 0] = (
            atom_site["Cartn_x"]
            .as_array(np.float32)
            .reshape((model_count, model_length))
        )
        atoms.coord[:, :, 1] = (
            atom_site["Cartn_y"]
            .as_array(np.float32)
            .reshape((model_count, model_length))
        )
        atoms.coord[:, :, 2] = (
            atom_site["Cartn_z"]
            .as_array(np.float32)
            .reshape((model_count, model_length))
        )

        box = _get_box(block)
        if box is not None:
            # Duplicate same box for each model
            atoms.box = np.repeat(box[np.newaxis, ...], model_count, axis=0)

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

        model_atom_site = _filter_model(atom_site, model)
        # Any field of the category would work here to get the length
        model_length = model_atom_site.row_count
        atoms = AtomArray(model_length)

        atoms.coord[:, 0] = model_atom_site["Cartn_x"].as_array(np.float32)
        atoms.coord[:, 1] = model_atom_site["Cartn_y"].as_array(np.float32)
        atoms.coord[:, 2] = model_atom_site["Cartn_z"].as_array(np.float32)

        atoms.box = _get_box(block)

    # The below part is the same for both, AtomArray and AtomArrayStack
    _fill_annotations(atoms, model_atom_site, extra_fields, use_author_fields)

    atoms, altloc_filtered_atom_site = _filter_altloc(atoms, model_atom_site, altloc)

    if include_bonds:
        if altloc == "all":
            raise ValueError(
                "Bond computation is not supported with `altloc='all', consider using "
                "'connect_via_residue_names()' afterwards"
            )

        if "chem_comp_bond" in block:
            try:
                custom_bond_dict = _parse_intra_residue_bonds(block["chem_comp_bond"])
            except KeyError:
                warnings.warn(
                    "The 'chem_comp_bond' category has missing columns, "
                    "falling back to using Chemical Component Dictionary",
                    UserWarning,
                )
                custom_bond_dict = None
            bonds = connect_via_residue_names(atoms, custom_bond_dict=custom_bond_dict)
        else:
            bonds = connect_via_residue_names(atoms)
        if "struct_conn" in block:
            bonds = bonds.merge(
                _parse_inter_residue_bonds(
                    altloc_filtered_atom_site,
                    block["struct_conn"],
                    atom_count=atoms.array_length(),
                )
            )
        atoms.bonds = bonds

    return atoms


def _get_block(pdbx_component, block_name):
    if not isinstance(pdbx_component, (CIFBlock, BinaryCIFBlock)):
        # Determine block
        if block_name is None:
            return pdbx_component.block
        else:
            return pdbx_component[block_name]
    else:
        return pdbx_component


def _get_or_fallback(category, key, fallback_key):
    """
    Return column related to key in category if it exists,
    otherwise try to get the column related to fallback key.
    """
    if key not in category:
        warnings.warn(
            f"Attribute '{key}' not found within 'atom_site' category. "
            f"The fallback attribute '{fallback_key}' will be used instead",
            UserWarning,
        )
        try:
            return category[fallback_key]
        except KeyError as key_exc:
            raise InvalidFileError(
                f"Fallback attribute '{fallback_key}' not found within "
                "'atom_site' category"
            ) from key_exc
    return category[key]


def _fill_annotations(array, atom_site, extra_fields, use_author_fields):
    """Fill atom_site annotations in atom array or atom array stack.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        Atom array or stack which will be annotated.
    atom_site : CIFCategory or BinaryCIFCategory
        ``atom_site`` category with values for one model.
    extra_fields : list of str
        Entry names, that are additionally added as annotation arrays.
    use_author_fields : bool
        Define if alternate fields prefixed with ``auth_`` should be used
        instead of ``label_``.
    """

    prefix, alt_prefix = ("auth", "label") if use_author_fields else ("label", "auth")

    array.set_annotation(
        "chain_id",
        _get_or_fallback(
            atom_site, f"{prefix}_asym_id", f"{alt_prefix}_asym_id"
        ).as_array(str),
    )
    array.set_annotation(
        "res_id",
        _get_or_fallback(
            atom_site, f"{prefix}_seq_id", f"{alt_prefix}_seq_id"
        ).as_array(int, -1),
    )
    array.set_annotation("ins_code", atom_site["pdbx_PDB_ins_code"].as_array(str, ""))
    array.set_annotation(
        "res_name",
        _get_or_fallback(
            atom_site, f"{prefix}_comp_id", f"{alt_prefix}_comp_id"
        ).as_array(str),
    )
    array.set_annotation("hetero", atom_site["group_PDB"].as_array(str) == "HETATM")
    array.set_annotation(
        "atom_name",
        _get_or_fallback(
            atom_site, f"{prefix}_atom_id", f"{alt_prefix}_atom_id"
        ).as_array(str),
    )
    array.set_annotation("element", atom_site["type_symbol"].as_array(str))

    if "atom_id" in extra_fields:
        if "id" in atom_site:
            array.set_annotation("atom_id", atom_site["id"].as_array(int))
        else:
            warnings.warn(
                "Missing 'id' in 'atom_site' category. 'atom_id' generated automatically.",
                UserWarning,
            )
            array.set_annotation("atom_id", np.arange(array.array_length()))
        extra_fields.remove("atom_id")
    if "b_factor" in extra_fields:
        if "B_iso_or_equiv" in atom_site:
            array.set_annotation(
                "b_factor", atom_site["B_iso_or_equiv"].as_array(float)
            )
        else:
            warnings.warn(
                "Missing 'B_iso_or_equiv' in 'atom_site' category. 'b_factor' will be set to `nan`.",
                UserWarning,
            )
            array.set_annotation("b_factor", np.full(array.array_length(), np.nan))
        extra_fields.remove("b_factor")
    if "occupancy" in extra_fields:
        if "occupancy" in atom_site:
            array.set_annotation("occupancy", atom_site["occupancy"].as_array(float))
        else:
            warnings.warn(
                "Missing 'occupancy' in 'atom_site' category. 'occupancy' will be assumed to be 1.0",
                UserWarning,
            )
            array.set_annotation(
                "occupancy", np.ones(array.array_length(), dtype=float)
            )
        extra_fields.remove("occupancy")
    if "charge" in extra_fields:
        if "pdbx_formal_charge" in atom_site:
            array.set_annotation(
                "charge",
                atom_site["pdbx_formal_charge"].as_array(
                    int, 0
                ),  # masked values are set to 0
            )
        else:
            warnings.warn(
                "Missing 'pdbx_formal_charge' in 'atom_site' category. 'charge' will be set to 0",
                UserWarning,
            )
            array.set_annotation("charge", np.zeros(array.array_length(), dtype=int))
        extra_fields.remove("charge")

    # Handle all remaining custom fields
    for field in extra_fields:
        array.set_annotation(field, atom_site[field].as_array(str))


def _parse_intra_residue_bonds(chem_comp_bond):
    """
    Create a :func:`connect_via_residue_names()` compatible
    `custom_bond_dict` from the ``chem_comp_bond`` category.
    """
    custom_bond_dict = {}
    for res_name, atom_1, atom_2, order, aromatic_flag in zip(
        chem_comp_bond["comp_id"].as_array(str),
        chem_comp_bond["atom_id_1"].as_array(str),
        chem_comp_bond["atom_id_2"].as_array(str),
        chem_comp_bond["value_order"].as_array(str),
        chem_comp_bond["pdbx_aromatic_flag"].as_array(str),
    ):
        if res_name not in custom_bond_dict:
            custom_bond_dict[res_name] = {}
        bond_type = COMP_BOND_ORDER_TO_TYPE.get(
            (order.upper(), aromatic_flag), BondType.ANY
        )
        custom_bond_dict[res_name][atom_1.item(), atom_2.item()] = bond_type
    return custom_bond_dict


def _parse_inter_residue_bonds(atom_site, struct_conn, atom_count=None):
    """
    Create inter-residue bonds by parsing the ``struct_conn`` category.
    The atom indices of each bond are found by matching the bond labels
    to the ``atom_site`` category.
    If atom_count is None, it will be inferred from the ``atom_site`` category.
    """
    # Identity symmetry operation
    IDENTITY = "1_555"
    # Columns in 'atom_site' that should be matched by 'struct_conn'
    COLUMNS = [
        "label_asym_id",
        "label_comp_id",
        "label_seq_id",
        "label_atom_id",
        "label_alt_id",
        "auth_asym_id",
        "auth_comp_id",
        "auth_seq_id",
        "pdbx_PDB_ins_code",
    ]

    covale_mask = np.isin(
        struct_conn["conn_type_id"].as_array(str),
        list(PDBX_BOND_TYPE_ID_TO_TYPE.keys()),
    )
    if "ptnr1_symmetry" in struct_conn:
        covale_mask &= struct_conn["ptnr1_symmetry"].as_array(str, IDENTITY) == IDENTITY
    if "ptnr2_symmetry" in struct_conn:
        covale_mask &= struct_conn["ptnr2_symmetry"].as_array(str, IDENTITY) == IDENTITY

    atom_indices = [None] * 2
    for i in range(2):
        reference_arrays = []
        query_arrays = []
        for col_name in COLUMNS:
            struct_conn_col_name = _get_struct_conn_col_name(col_name, i + 1)
            if col_name not in atom_site or struct_conn_col_name not in struct_conn:
                continue
            # Ensure both arrays have the same dtype to allow comparison
            reference = atom_site[col_name].as_array()
            dtype = reference.dtype
            query = struct_conn[struct_conn_col_name].as_array(dtype)
            if np.issubdtype(reference.dtype, str):
                # The mask value is not necessarily consistent
                # between query and reference
                # -> make it consistent
                reference[reference == "?"] = "."
                query[query == "?"] = "."
            reference_arrays.append(reference)
            query_arrays.append(query[covale_mask])
        # Match the combination of 'label_asym_id', 'label_comp_id', etc.
        # in 'atom_site' and 'struct_conn'
        atom_indices[i] = _find_matches(query_arrays, reference_arrays)
    atoms_indices_1 = atom_indices[0]
    atoms_indices_2 = atom_indices[1]

    # Some bonds in 'struct_conn' may not be found in 'atom_site'
    # This is okay,
    # as 'atom_site' might already be reduced to a single model
    mapping_exists_mask = (atoms_indices_1 != -1) & (atoms_indices_2 != -1)
    atoms_indices_1 = atoms_indices_1[mapping_exists_mask]
    atoms_indices_2 = atoms_indices_2[mapping_exists_mask]

    bond_type_id = struct_conn["conn_type_id"].as_array()
    # Consecutively apply the same masks as applied to the atom indices
    # Logical combination does not work here,
    # as the second mask was created based on already filtered data
    bond_type_id = bond_type_id[covale_mask][mapping_exists_mask]
    # The type ID is always present in the dictionary,
    # as it was used to filter the applicable bonds
    bond_types = [PDBX_BOND_TYPE_ID_TO_TYPE[type_id] for type_id in bond_type_id]

    return BondList(
        atom_count if atom_count is not None else atom_site.row_count,
        np.stack([atoms_indices_1, atoms_indices_2, bond_types], axis=-1),
    )


def _find_matches(query_arrays, reference_arrays):
    """
    For each index in the `query_arrays` find the indices in the
    `reference_arrays` where all query values match the reference counterpart.
    If no match is found for a query, the corresponding index is -1.
    """
    if (
        query_arrays[0].shape[0] * reference_arrays[0].shape[0]
        <= FIND_MATCHES_SWITCH_THRESHOLD
    ):
        match_indices = _find_matches_by_dense_array(query_arrays, reference_arrays)
    else:
        match_indices = _find_matches_by_dict(query_arrays, reference_arrays)
    return match_indices


def _find_matches_by_dense_array(query_arrays, reference_arrays):
    match_masks_for_all_columns = np.stack(
        [
            query[:, np.newaxis] == reference[np.newaxis, :]
            for query, reference in zip(query_arrays, reference_arrays)
        ],
        axis=-1,
    )
    match_masks = np.all(match_masks_for_all_columns, axis=-1)
    query_matches, reference_matches = np.where(match_masks)

    # Duplicate matches indicate that an atom from the query cannot
    # be uniquely matched to an atom in the reference
    unique_query_matches, counts = np.unique(query_matches, return_counts=True)
    if np.any(counts > 1):
        ambiguous_query = unique_query_matches[np.where(counts > 1)[0][0]]
        raise InvalidFileError(
            f"The covalent bond in the 'struct_conn' category at index "
            f"{ambiguous_query} cannot be unambiguously assigned to atoms in "
            f"the 'atom_site' category"
        )

    # -1 indicates that no match was found in the reference
    match_indices = np.full(len(query_arrays[0]), -1, dtype=int)
    match_indices[query_matches] = reference_matches
    return match_indices


def _find_matches_by_dict(query_arrays, reference_arrays):
    # Convert reference arrays to a dictionary for O(1) lookups
    reference_dict = {}
    ambiguous_keys = set()
    for ref_idx, ref_row in enumerate(zip(*reference_arrays)):
        ref_key = tuple(ref_row)
        if ref_key in reference_dict:
            ambiguous_keys.add(ref_key)
            continue
        reference_dict[ref_key] = ref_idx

    match_indices = []
    for query_idx, query_row in enumerate(zip(*query_arrays)):
        query_key = tuple(query_row)
        occurrence = reference_dict.get(query_key)

        if occurrence is None:
            # -1 indicates that no match was found in the reference
            match_indices.append(-1)
        elif query_key in ambiguous_keys:
            # The query cannot be uniquely matched to an atom in the reference
            raise InvalidFileError(
                f"The covalent bond in the 'struct_conn' category at index "
                f"{query_idx} cannot be unambiguously assigned to atoms in "
                f"the 'atom_site' category"
            )
        else:
            match_indices.append(occurrence)

    return np.array(match_indices)


def _get_struct_conn_col_name(col_name, partner):
    """
    For a column name in ``atom_site`` get the corresponding column name
    in ``struct_conn``.
    """
    if col_name == "label_alt_id":
        return f"pdbx_ptnr{partner}_label_alt_id"
    elif col_name.startswith("pdbx_"):
        # Move 'pdbx_' to front
        return f"pdbx_ptnr{partner}_{col_name[5:]}"
    else:
        return f"ptnr{partner}_{col_name}"


def _filter_altloc(array, atom_site, altloc):
    """
    Filter the given :class:`AtomArray` and ``atom_site`` category to the rows
    specified by the given *altloc* identifier.
    """
    altloc_ids = atom_site.get("label_alt_id")
    occupancy = atom_site.get("occupancy")

    if altloc == "all":
        array.set_annotation("altloc_id", altloc_ids.as_array(str))
        return array, atom_site
    elif altloc_ids is None or (altloc_ids.mask.array != MaskValue.PRESENT).all():
        # No altlocs in atom_site category
        return array, atom_site
    elif altloc == "occupancy" and occupancy is not None:
        mask = filter_highest_occupancy_altloc(
            array, altloc_ids.as_array(str), occupancy.as_array(float)
        )
        return array[..., mask], _filter(atom_site, mask)
    # 'first' is also fallback if file has no occupancy information
    elif altloc == "first":
        mask = filter_first_altloc(array, altloc_ids.as_array(str))
        return array[..., mask], _filter(atom_site, mask)
    else:
        raise ValueError(f"'{altloc}' is not a valid 'altloc' option")


def _filter_model(atom_site, model):
    """
    Reduce the ``atom_site`` category to the values for the given
    model.

    Parameters
    ----------
    atom_site : CIFCategory or BinaryCIFCategory
        ``atom_site`` category containing all models.
    model : int
        The model to be selected.

    Returns
    -------
    atom_site : CIFCategory or BinaryCIFCategory
        The ``atom_site`` category containing only the selected model.
    """
    models = atom_site["pdbx_PDB_model_num"].as_array(np.int32)
    _, model_starts = np.unique(models, return_index=True)
    model_starts.sort()
    # Append exclusive stop
    model_starts = np.append(model_starts, [atom_site.row_count])
    # Indexing starts at 0, but model number starts at 1
    model_index = model - 1
    index = slice(model_starts[model_index], model_starts[model_index + 1])
    return _filter(atom_site, index)


def _get_box(block):
    cell = block.get("cell")
    if cell is None:
        return None
    try:
        len_a, len_b, len_c = [
            float(cell[length].as_item())
            for length in ["length_a", "length_b", "length_c"]
        ]
        alpha, beta, gamma = [
            np.deg2rad(float(cell[angle].as_item()))
            for angle in ["angle_alpha", "angle_beta", "angle_gamma"]
        ]
    except ValueError:
        # 'cell_dict' has no proper unit cell values, e.g. '?'
        return None
    return vectors_from_unitcell(len_a, len_b, len_c, alpha, beta, gamma)


def set_structure(
    pdbx_file,
    array,
    data_block=None,
    include_bonds=False,
    extra_fields=[],
):
    """
    Set the ``atom_site`` category with atom information from an
    :class:`AtomArray` or :class:`AtomArrayStack`.

    This will save the coordinates, the mandatory annotation categories
    and the optional annotation categories
    ``atom_id``, ``b_factor``, ``occupancy`` and ``charge``.
    If the atom array (stack) contains the annotation ``'atom_id'``,
    these values will be used for atom numbering instead of continuous
    numbering.
    Furthermore, inter-residue bonds will be written into the
    ``struct_conn`` category.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    array : AtomArray or AtomArrayStack
        The structure to be written. If a stack is given, each array in
        the stack will be in a separate model.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
        If the file is empty, a new data block will be created.
    include_bonds : bool, optional
        If set to true and `array` has associated ``bonds`` , the
        intra-residue bonds will be written into the ``chem_comp_bond``
        category.
        Inter-residue bonds will be written into the ``struct_conn``
        independent of this parameter.
    extra_fields : list of str, optional
        List of additional fields from the ``atom_site`` category
        that should be written into the file.
        Default is an empty list.

    Notes
    -----
    In some cases, the written inter-residue bonds cannot be read again
    due to ambiguity to which atoms the bond refers.
    This is the case, when two equal residues in the same chain have
    the same (or a masked) `res_id`.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile()
    >>> set_structure(file, atom_array)
    >>> file.write(os.path.join(path_to_directory, "structure.cif"))
    """
    _check_non_empty(array)

    block = _get_or_create_block(pdbx_file, data_block)
    Category = block.subcomponent_class()
    Column = Category.subcomponent_class()

    # Fill PDBx columns from information
    # in structures' attribute arrays as good as possible
    atom_site = Category()
    atom_site["group_PDB"] = np.where(array.hetero, "HETATM", "ATOM")
    atom_site["type_symbol"] = np.copy(array.element)
    atom_site["label_atom_id"] = np.copy(array.atom_name)
    atom_site["label_alt_id"] = Column(
        # AtomArrays do not store altloc atoms
        np.full(array.array_length(), "."),
        np.full(array.array_length(), MaskValue.INAPPLICABLE),
    )
    atom_site["label_comp_id"] = np.copy(array.res_name)
    atom_site["label_asym_id"] = np.copy(array.chain_id)
    atom_site["label_entity_id"] = (
        np.copy(array.label_entity_id)
        if "label_entity_id" in array.get_annotation_categories()
        else _determine_entity_id(array.chain_id)
    )
    atom_site["label_seq_id"] = np.copy(array.res_id)
    atom_site["pdbx_PDB_ins_code"] = Column(
        np.copy(array.ins_code),
        np.where(array.ins_code == "", MaskValue.INAPPLICABLE, MaskValue.PRESENT),
    )
    atom_site["auth_seq_id"] = atom_site["label_seq_id"]
    atom_site["auth_comp_id"] = atom_site["label_comp_id"]
    atom_site["auth_asym_id"] = atom_site["label_asym_id"]
    atom_site["auth_atom_id"] = atom_site["label_atom_id"]

    annot_categories = array.get_annotation_categories()
    if "atom_id" in annot_categories:
        atom_site["id"] = np.copy(array.atom_id)
    if "b_factor" in annot_categories:
        atom_site["B_iso_or_equiv"] = np.copy(array.b_factor)
    if "occupancy" in annot_categories:
        atom_site["occupancy"] = np.copy(array.occupancy)
    if "charge" in annot_categories:
        atom_site["pdbx_formal_charge"] = Column(
            np.array([f"{c:+d}" if c != 0 else "?" for c in array.charge]),
            np.where(array.charge == 0, MaskValue.MISSING, MaskValue.PRESENT),
        )

    # Handle all remaining custom fields
    if len(extra_fields) > 0:
        # ... check to avoid clashes with standard annotations
        _standard_annotations = [
            "hetero",
            "element",
            "atom_name",
            "res_name",
            "chain_id",
            "res_id",
            "ins_code",
            "atom_id",
            "b_factor",
            "occupancy",
            "charge",
        ]
        _reserved_annotation_names = list(atom_site.keys()) + _standard_annotations

        for annot in extra_fields:
            if annot in _reserved_annotation_names:
                raise ValueError(
                    f"Annotation name '{annot}' is reserved and cannot be written to as extra field. "
                    "Please choose another name."
                )
            atom_site[annot] = np.copy(array.get_annotation(annot))

    if array.bonds is not None:
        struct_conn = _set_inter_residue_bonds(array, atom_site)
        if struct_conn is not None:
            block["struct_conn"] = struct_conn
        if include_bonds:
            chem_comp_bond = _set_intra_residue_bonds(array, atom_site)
            if chem_comp_bond is not None:
                block["chem_comp_bond"] = chem_comp_bond

    # In case of a single model handle each coordinate
    # simply like a flattened array
    if isinstance(array, AtomArray) or (
        isinstance(array, AtomArrayStack) and array.stack_depth() == 1
    ):
        # 'ravel' flattens coord without copy
        # in case of stack with stack_depth = 1
        atom_site["Cartn_x"] = np.copy(np.ravel(array.coord[..., 0]))
        atom_site["Cartn_y"] = np.copy(np.ravel(array.coord[..., 1]))
        atom_site["Cartn_z"] = np.copy(np.ravel(array.coord[..., 2]))
        atom_site["pdbx_PDB_model_num"] = np.ones(array.array_length(), dtype=np.int32)
    # In case of multiple models repeat annotations
    # and use model specific coordinates
    else:
        atom_site = _repeat(atom_site, array.stack_depth())
        coord = np.reshape(array.coord, (array.stack_depth() * array.array_length(), 3))
        atom_site["Cartn_x"] = np.copy(coord[:, 0])
        atom_site["Cartn_y"] = np.copy(coord[:, 1])
        atom_site["Cartn_z"] = np.copy(coord[:, 2])
        atom_site["pdbx_PDB_model_num"] = np.repeat(
            np.arange(1, array.stack_depth() + 1, dtype=np.int32),
            repeats=array.array_length(),
        )
    if "atom_id" not in annot_categories:
        # Count from 1
        atom_site["id"] = np.arange(1, len(atom_site["group_PDB"]) + 1)
    block["atom_site"] = atom_site

    # Write box into file
    if array.box is not None:
        # PDBx files can only store one box for all models
        # -> Use first box
        if array.box.ndim == 3:
            box = array.box[0]
        else:
            box = array.box
        len_a, len_b, len_c, alpha, beta, gamma = unitcell_from_vectors(box)
        cell = Category()
        cell["length_a"] = len_a
        cell["length_b"] = len_b
        cell["length_c"] = len_c
        cell["angle_alpha"] = np.rad2deg(alpha)
        cell["angle_beta"] = np.rad2deg(beta)
        cell["angle_gamma"] = np.rad2deg(gamma)
        block["cell"] = cell


def _check_non_empty(array):
    if isinstance(array, AtomArray):
        if array.array_length() == 0:
            raise BadStructureError("Structure must not be empty")
    elif isinstance(array, AtomArrayStack):
        if array.array_length() == 0 or array.stack_depth() == 0:
            raise BadStructureError("Structure must not be empty")
    else:
        raise ValueError(
            "Structure must be AtomArray or AtomArrayStack, "
            f"but got {type(array).__name__}"
        )


def _get_or_create_block(pdbx_component, block_name):
    Block = pdbx_component.subcomponent_class()

    if isinstance(pdbx_component, (CIFFile, BinaryCIFFile)):
        if block_name is None:
            if len(pdbx_component) > 0:
                block_name = next(iter(pdbx_component.keys()))
            else:
                # File is empty -> invent a new block name
                block_name = "structure"

        if block_name not in pdbx_component:
            block = Block()
            pdbx_component[block_name] = block
        return pdbx_component[block_name]
    else:
        # Already a block
        return pdbx_component


def _determine_entity_id(chain_id):
    entity_id = np.zeros(len(chain_id), dtype=int)
    # Dictionary that translates chain_id to entity_id
    id_translation = {}
    id = 1
    for i in range(len(chain_id)):
        try:
            entity_id[i] = id_translation[chain_id[i]]
        except KeyError:
            # chain_id is not in dictionary -> new entry
            id_translation[chain_id[i]] = id
            entity_id[i] = id_translation[chain_id[i]]
            id += 1
    return entity_id


def _repeat(category, repetitions):
    Category = type(category)
    Column = Category.subcomponent_class()
    Data = Column.subcomponent_class()

    category_dict = {}
    for key, column in category.items():
        if isinstance(column, BinaryCIFColumn):
            data_encoding = column.data.encoding
            # Optimization: The repeated string array has the same
            # unique values, as the original string array
            # -> Use same unique values (faster due to shorter array)
            if isinstance(data_encoding[0], StringArrayEncoding):
                data_encoding[0].strings = np.unique(column.data.array)
            data = Data(np.tile(column.data.array, repetitions), data_encoding)
        else:
            data = Data(np.tile(column.data.array, repetitions))
        mask = (
            Data(np.tile(column.mask.array, repetitions))
            if column.mask is not None
            else None
        )
        category_dict[key] = Column(data, mask)
    return Category(category_dict)


def _set_intra_residue_bonds(array, atom_site):
    """
    Create the ``chem_comp_bond`` category containing the intra-residue
    bonds.
    ``atom_site`` is only used to infer the right :class:`Category` type
    (either :class:`CIFCategory` or :class:`BinaryCIFCategory`).
    """
    if (array.res_name == "").any():
        raise BadStructureError(
            "Structure contains atoms with empty residue name, "
            "but it is required to write intra-residue bonds"
        )
    if (array.atom_name == "").any():
        raise BadStructureError(
            "Structure contains atoms with empty atom name, "
            "but it is required to write intra-residue bonds"
        )

    Category = type(atom_site)
    Column = Category.subcomponent_class()

    bond_array = _filter_bonds(array, "intra")
    if len(bond_array) == 0:
        return None
    value_order = np.zeros(len(bond_array), dtype="U4")
    aromatic_flag = np.zeros(len(bond_array), dtype="U1")
    for i, bond_type in enumerate(bond_array[:, 2]):
        if bond_type == BondType.ANY:
            # ANY bonds will be masked anyway, no need to set the value
            continue
        order, aromatic = COMP_BOND_TYPE_TO_ORDER[bond_type]
        value_order[i] = order
        aromatic_flag[i] = aromatic
    any_mask = bond_array[:, 2] == BondType.ANY

    # Remove already existing residue and atom name combinations
    # These appear when the structure contains a residue multiple times
    atom_id_1 = array.atom_name[bond_array[:, 0]]
    atom_id_2 = array.atom_name[bond_array[:, 1]]
    # Take the residue name from the first atom index, as the residue
    # name is the same for both atoms, since we have only intra bonds
    comp_id = array.res_name[bond_array[:, 0]]
    _, unique_indices = np.unique(
        np.stack([comp_id, atom_id_1, atom_id_2], axis=-1), axis=0, return_index=True
    )
    unique_indices.sort()

    chem_comp_bond = Category()
    n_bonds = len(unique_indices)
    chem_comp_bond["pdbx_ordinal"] = np.arange(1, n_bonds + 1, dtype=np.int32)
    chem_comp_bond["comp_id"] = comp_id[unique_indices]
    chem_comp_bond["atom_id_1"] = atom_id_1[unique_indices]
    chem_comp_bond["atom_id_2"] = atom_id_2[unique_indices]
    chem_comp_bond["value_order"] = Column(
        value_order[unique_indices],
        np.where(any_mask[unique_indices], MaskValue.MISSING, MaskValue.PRESENT),
    )
    chem_comp_bond["pdbx_aromatic_flag"] = Column(
        aromatic_flag[unique_indices],
        np.where(any_mask[unique_indices], MaskValue.MISSING, MaskValue.PRESENT),
    )
    # BondList does not contain stereo information
    # -> all values are missing
    chem_comp_bond["pdbx_stereo_config"] = Column(
        np.zeros(n_bonds, dtype="U1"),
        np.full(n_bonds, MaskValue.MISSING),
    )
    return chem_comp_bond


def _set_inter_residue_bonds(array, atom_site):
    """
    Create the ``struct_conn`` category containing the inter-residue
    bonds.
    The involved atoms are identified by annotations from the
    ``atom_site`` category.
    """
    COLUMNS = [
        "label_asym_id",
        "label_comp_id",
        "label_seq_id",
        "label_atom_id",
        "pdbx_PDB_ins_code",
    ]

    Category = type(atom_site)
    Column = Category.subcomponent_class()

    bond_array = _filter_bonds(array, "inter")
    if len(bond_array) == 0:
        return None

    # Filter out 'standard' links, i.e. backbone bonds between adjacent canonical
    # nucleotide/amino acid residues
    bond_array = bond_array[~_filter_canonical_links(array, bond_array)]
    if len(bond_array) == 0:
        return None

    struct_conn = Category()
    struct_conn["id"] = np.arange(1, len(bond_array) + 1)
    struct_conn["conn_type_id"] = [
        PDBX_BOND_TYPE_TO_TYPE_ID[btype] for btype in bond_array[:, 2]
    ]
    struct_conn["pdbx_value_order"] = Column(
        np.array([PDBX_BOND_TYPE_TO_ORDER[btype] for btype in bond_array[:, 2]]),
        np.where(
            np.isin(bond_array[:, 2], (BondType.ANY, BondType.COORDINATION)),
            MaskValue.MISSING,
            MaskValue.PRESENT,
        ),
    )
    # Write the identifying annotation...
    for col_name in COLUMNS:
        annot = atom_site[col_name].as_array()
        # ...for each bond partner
        for i in range(2):
            atom_indices = bond_array[:, i]
            struct_conn[_get_struct_conn_col_name(col_name, i + 1)] = annot[
                atom_indices
            ]
    return struct_conn


def _filter_bonds(array, connection):
    """
    Get a bonds array, that contain either only intra-residue or
    only inter-residue bonds.
    """
    bond_array = array.bonds.as_array()
    # To save computation time call 'get_residue_starts_for()' only once
    # with indices of the first and second atom of each bond
    residue_starts_1, residue_starts_2 = (
        get_residue_starts_for(array, bond_array[:, :2].flatten()).reshape(-1, 2).T
    )
    if connection == "intra":
        return bond_array[residue_starts_1 == residue_starts_2]
    elif connection == "inter":
        return bond_array[residue_starts_1 != residue_starts_2]
    else:
        raise ValueError("Invalid 'connection' option")


def _filter_canonical_links(array, bond_array):
    """
    Filter out peptide bonds between adjacent canonical amino acid residues.
    """
    # Get the residue index for each bonded atom
    residue_indices = get_residue_positions(array, bond_array[:, :2].flatten()).reshape(
        -1, 2
    )

    return (
        # Must be canonical residues
        np.isin(array.res_name[bond_array[:, 0]], CANONICAL_RESIDUE_LIST) &
        np.isin(array.res_name[bond_array[:, 1]], CANONICAL_RESIDUE_LIST) &
        # Must be backbone bond
        np.isin(array.atom_name[bond_array[:, 0]], ("C", "O3'")) &
        np.isin(array.atom_name[bond_array[:, 1]], ("N", "P")) &
        # Must connect adjacent residues
        residue_indices[:, 1] - residue_indices[:, 0] == 1
    )  # fmt: skip


def get_component(
    pdbx_file,
    data_block=None,
    use_ideal_coord=True,
    res_name=None,
    allow_missing_coord=False,
):
    """
    Create an :class:`AtomArray` for a chemical component from the
    ``chem_comp_atom`` and, if available, the ``chem_comp_bond``
    category in a file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
    use_ideal_coord : bool, optional
        If true, the *ideal* coordinates are read from the file
        (``pdbx_model_Cartn_<dim>_ideal`` fields), typically
        originating from computations.
        If set to false, alternative coordinates are read
        (``model_Cartn_<dim>_`` fields).
    res_name : str
        In rare cases the categories may contain rows for multiple
        components.
        In this case, the component with the given residue name is
        read.
        By default, all rows would be read in this case.
    allow_missing_coord : bool, optional
        Whether to allow missing coordinate values in components.
        If ``True``, these will be represented as ``nan`` values.
        If ``False``, a ``ValueError`` is raised when missing coordinates
        are encountered.

    Returns
    -------
    array : AtomArray
        The parsed chemical component.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(
    ...     os.path.join(path_to_structures, "molecules", "TYR.cif")
    ... )
    >>> comp = get_component(file)
    >>> print(comp)
    HET         0  TYR N      N         1.320    0.952    1.428
    HET         0  TYR CA     C        -0.018    0.429    1.734
    HET         0  TYR C      C        -0.103    0.094    3.201
    HET         0  TYR O      O         0.886   -0.254    3.799
    HET         0  TYR CB     C        -0.274   -0.831    0.907
    HET         0  TYR CG     C        -0.189   -0.496   -0.559
    HET         0  TYR CD1    C         1.022   -0.589   -1.219
    HET         0  TYR CD2    C        -1.324   -0.102   -1.244
    HET         0  TYR CE1    C         1.103   -0.282   -2.563
    HET         0  TYR CE2    C        -1.247    0.210   -2.587
    HET         0  TYR CZ     C        -0.032    0.118   -3.252
    HET         0  TYR OH     O         0.044    0.420   -4.574
    HET         0  TYR OXT    O        -1.279    0.184    3.842
    HET         0  TYR H      H         1.977    0.225    1.669
    HET         0  TYR H2     H         1.365    1.063    0.426
    HET         0  TYR HA     H        -0.767    1.183    1.489
    HET         0  TYR HB2    H         0.473   -1.585    1.152
    HET         0  TYR HB3    H        -1.268   -1.219    1.134
    HET         0  TYR HD1    H         1.905   -0.902   -0.683
    HET         0  TYR HD2    H        -2.269   -0.031   -0.727
    HET         0  TYR HE1    H         2.049   -0.354   -3.078
    HET         0  TYR HE2    H        -2.132    0.523   -3.121
    HET         0  TYR HH     H        -0.123   -0.399   -5.059
    HET         0  TYR HXT    H        -1.333   -0.030    4.784
    """
    block = _get_block(pdbx_file, data_block)

    try:
        atom_category = block["chem_comp_atom"]
    except KeyError:
        raise InvalidFileError("Missing 'chem_comp_atom' category in file")
    if res_name is not None:
        atom_category = _filter(
            atom_category, atom_category["comp_id"].as_array() == res_name
        )
        if atom_category.row_count == 0:
            raise KeyError(
                f"No rows with residue name '{res_name}' found in "
                f"'chem_comp_atom' category"
            )

    array = AtomArray(atom_category.row_count)

    array.set_annotation("hetero", np.full(len(atom_category["comp_id"]), True))
    array.set_annotation("res_name", atom_category["comp_id"].as_array(str))
    array.set_annotation("atom_name", atom_category["atom_id"].as_array(str))
    array.set_annotation("element", atom_category["type_symbol"].as_array(str))
    array.set_annotation("charge", atom_category["charge"].as_array(int, 0))

    coord_fields = [f"pdbx_model_Cartn_{dim}_ideal" for dim in ("x", "y", "z")]
    alt_coord_fields = [f"model_Cartn_{dim}" for dim in ("x", "y", "z")]
    if not use_ideal_coord:
        # Swap with the fallback option
        coord_fields, alt_coord_fields = alt_coord_fields, coord_fields
    try:
        array.coord = _parse_component_coordinates(
            [atom_category[field] for field in coord_fields]
        )
    except Exception as err:
        if isinstance(err, KeyError):
            key = err.args[0]
            warnings.warn(
                f"Attribute '{key}' not found within 'chem_comp_atom' category. "
                f"The fallback coordinates will be used instead",
                UserWarning,
            )
        elif isinstance(err, ValueError):
            warnings.warn(
                "The coordinates are missing for some atoms. "
                "The fallback coordinates will be used instead",
                UserWarning,
            )
        else:
            raise
        array.coord = _parse_component_coordinates(
            [atom_category[field] for field in alt_coord_fields],
            allow_missing=allow_missing_coord,
        )

    try:
        bond_category = block["chem_comp_bond"]
        if res_name is not None:
            bond_category = _filter(
                bond_category, bond_category["comp_id"].as_array() == res_name
            )
    except KeyError:
        warnings.warn(
            "Category 'chem_comp_bond' not found. No bonds will be parsed",
            UserWarning,
        )
    else:
        bonds = BondList(array.array_length())
        for atom1, atom2, order, aromatic_flag in zip(
            bond_category["atom_id_1"].as_array(str),
            bond_category["atom_id_2"].as_array(str),
            bond_category["value_order"].as_array(str),
            bond_category["pdbx_aromatic_flag"].as_array(str),
        ):
            atom_i = np.where(array.atom_name == atom1)[0][0]
            atom_j = np.where(array.atom_name == atom2)[0][0]
            bond_type = COMP_BOND_ORDER_TO_TYPE[order, aromatic_flag]
            bonds.add_bond(atom_i, atom_j, bond_type)
        array.bonds = bonds

    return array


def _parse_component_coordinates(coord_columns, allow_missing=False):
    coord = np.zeros((len(coord_columns[0]), 3), dtype=np.float32)
    for i, column in enumerate(coord_columns):
        if column.mask is not None and column.mask.array.any():
            if allow_missing:
                warnings.warn(
                    "Missing coordinates for some atoms. Those will be set to nan",
                    UserWarning,
                )
            else:
                raise ValueError(
                    "Missing coordinates for some atoms",
                )
        coord[:, i] = column.as_array(np.float32, masked_value=np.nan)
    return coord


def set_component(pdbx_file, array, data_block=None):
    """
    Set the ``chem_comp_atom`` and, if bonds are available,
    ``chem_comp_bond`` category with atom information from an
    :class:`AtomArray`.

    This will save the coordinates, the mandatory annotation categories
    and the optional ``charge`` category as well as an associated
    :class:`BondList`, if available.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    array : AtomArray
        The chemical component to be written.
        Must contain only a single residue.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the file is empty, a new data will be created.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
    """
    _check_non_empty(array)

    block = _get_or_create_block(pdbx_file, data_block)
    Category = block.subcomponent_class()

    if get_residue_count(array) > 1:
        raise BadStructureError("The input atom array must comprise only one residue")
    res_name = array.res_name[0]

    annot_categories = array.get_annotation_categories()
    if "charge" in annot_categories:
        charge = array.charge.astype("U2")
    else:
        charge = np.full(array.array_length(), "?", dtype="U2")

    atom_cat = Category()
    atom_cat["comp_id"] = np.full(array.array_length(), res_name)
    atom_cat["atom_id"] = np.copy(array.atom_name)
    atom_cat["alt_atom_id"] = atom_cat["atom_id"]
    atom_cat["type_symbol"] = np.copy(array.element)
    atom_cat["charge"] = charge
    atom_cat["model_Cartn_x"] = np.copy(array.coord[:, 0])
    atom_cat["model_Cartn_y"] = np.copy(array.coord[:, 1])
    atom_cat["model_Cartn_z"] = np.copy(array.coord[:, 2])
    atom_cat["pdbx_model_Cartn_x_ideal"] = atom_cat["model_Cartn_x"]
    atom_cat["pdbx_model_Cartn_y_ideal"] = atom_cat["model_Cartn_y"]
    atom_cat["pdbx_model_Cartn_z_ideal"] = atom_cat["model_Cartn_z"]
    atom_cat["pdbx_component_atom_id"] = atom_cat["atom_id"]
    atom_cat["pdbx_component_comp_id"] = atom_cat["comp_id"]
    atom_cat["pdbx_ordinal"] = np.arange(1, array.array_length() + 1).astype(str)
    block["chem_comp_atom"] = atom_cat

    if array.bonds is not None and array.bonds.get_bond_count() > 0:
        bond_array = array.bonds.as_array()
        order_flags = []
        aromatic_flags = []
        for bond_type in bond_array[:, 2]:
            order_flag, aromatic_flag = COMP_BOND_TYPE_TO_ORDER[bond_type]
            order_flags.append(order_flag)
            aromatic_flags.append(aromatic_flag)

        bond_cat = Category()
        bond_cat["comp_id"] = np.full(len(bond_array), res_name)
        bond_cat["atom_id_1"] = array.atom_name[bond_array[:, 0]]
        bond_cat["atom_id_2"] = array.atom_name[bond_array[:, 1]]
        bond_cat["value_order"] = np.array(order_flags)
        bond_cat["pdbx_aromatic_flag"] = np.array(aromatic_flags)
        bond_cat["pdbx_ordinal"] = np.arange(1, len(bond_array) + 1).astype(str)
        block["chem_comp_bond"] = bond_cat


def list_assemblies(pdbx_file, data_block=None):
    """
    List the biological assemblies that are available for the structure
    in the given file.

    This function receives the data from the ``pdbx_struct_assembly``
    category in the file.
    Consequently, this category must be present in the file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.

    Returns
    -------
    assemblies : dict of str -> str
        A dictionary that maps an assembly ID to a description of the
        corresponding assembly.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1f2n.cif"))
    >>> assembly_ids = list_assemblies(file)
    >>> for key, val in assembly_ids.items():
    ...     print(f"'{key}' : '{val}'")
    '1' : 'complete icosahedral assembly'
    '2' : 'icosahedral asymmetric unit'
    '3' : 'icosahedral pentamer'
    '4' : 'icosahedral 23 hexamer'
    '5' : 'icosahedral asymmetric unit, std point frame'
    '6' : 'crystal asymmetric unit, crystal frame'
    """
    block = _get_block(pdbx_file, data_block)

    try:
        assembly_category = block["pdbx_struct_assembly"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_assembly' category")
    return {
        id: details
        for id, details in zip(
            assembly_category["id"].as_array(str),
            assembly_category["details"].as_array(str),
        )
    }


def get_assembly(
    pdbx_file,
    assembly_id=None,
    model=None,
    data_block=None,
    altloc="first",
    extra_fields=None,
    use_author_fields=True,
    include_bonds=False,
):
    """
    Build the given biological assembly.

    This function receives the data from the
    ``pdbx_struct_assembly_gen``, ``pdbx_struct_oper_list`` and
    ``atom_site`` categories in the file.
    Consequently, these categories must be present in the file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    assembly_id : str
        The assembly to build.
        Available assembly IDs can be obtained via
        :func:`list_assemblies()`.
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
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
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
        to other categories in the file, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    assembly : AtomArray or AtomArrayStack
        The assembly.
        The return type depends on the `model` parameter.
        Contains the `sym_id` annotation, which enumerates the copies of the asymmetric
        unit in the assembly.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1f2n.cif"))
    >>> assembly = get_assembly(file, model=1)
    """
    block = _get_block(pdbx_file, data_block)

    try:
        assembly_gen_category = block["pdbx_struct_assembly_gen"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_assembly_gen' category")

    try:
        struct_oper_category = block["pdbx_struct_oper_list"]
    except KeyError:
        raise InvalidFileError("File has no 'pdbx_struct_oper_list' category")

    assembly_ids = assembly_gen_category["assembly_id"].as_array(str)
    if assembly_id is None:
        assembly_id = assembly_ids[0]
    elif assembly_id not in assembly_ids:
        raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    ### Calculate all possible transformations
    transformations = _get_transformations(struct_oper_category)

    ### Get structure according to additional parameters
    # Include 'label_asym_id' as annotation array
    # for correct asym ID filtering
    extra_fields = [] if extra_fields is None else extra_fields
    if "label_asym_id" in extra_fields:
        extra_fields_and_asym = extra_fields
    else:
        # The operations apply on asym IDs
        # -> they need to be included to select the correct atoms
        extra_fields_and_asym = extra_fields + ["label_asym_id"]
    structure = get_structure(
        pdbx_file,
        model,
        data_block,
        altloc,
        extra_fields_and_asym,
        use_author_fields,
        include_bonds,
    )

    ### Get transformations and apply them to the affected asym IDs
    chain_ops = defaultdict(list)
    for id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].as_array(str),
        assembly_gen_category["oper_expression"].as_array(str),
        assembly_gen_category["asym_id_list"].as_array(str),
    ):
        # Find the operation expressions for given assembly ID
        # We already asserted that the ID is actually present
        if id == assembly_id:
            for chain_id in asym_id_expr.split(","):
                chain_ops[chain_id].extend(_parse_operation_expression(op_expr))

    sub_assemblies = []
    for asym_id, op_list in chain_ops.items():
        sub_struct = structure[..., structure.label_asym_id == asym_id]
        sub_assembly = _apply_transformations(sub_struct, transformations, op_list)
        # Merge the chain's sub_assembly into the rest of the assembly
        sub_assemblies.append(sub_assembly)
    assembly = concatenate(sub_assemblies)

    # Sort AtomArray or AtomArrayStack by 'sym_id'
    max_sym_id = assembly.sym_id.max()
    assembly = concatenate(
        [assembly[..., assembly.sym_id == sym_id] for sym_id in range(max_sym_id + 1)]
    )

    # Remove 'label_asym_id', if it was not included in the original
    # user-supplied 'extra_fields'
    if "label_asym_id" not in extra_fields:
        assembly.del_annotation("label_asym_id")

    return assembly


def _apply_transformations(structure, transformation_dict, operations):
    """
    Get subassembly by applying the given operations to the input
    structure containing affected asym IDs.
    """
    # Additional first dimesion for 'structure.repeat()'
    assembly_coord = np.zeros((len(operations),) + structure.coord.shape)
    # Apply corresponding transformation for each copy in the assembly
    for i, operation in enumerate(operations):
        coord = structure.coord
        # Execute for each transformation step
        # in the operation expression
        for op_step in operation:
            coord = transformation_dict[op_step].apply(coord)
        assembly_coord[i] = coord

    assembly = repeat(structure, assembly_coord)
    assembly.set_annotation(
        "sym_id", np.repeat(np.arange(len(operations)), structure.array_length())
    )
    return assembly


def _get_transformations(struct_oper):
    """
    Get affine transformation for each operation ID in ``pdbx_struct_oper_list``.
    """
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [
                [
                    struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index]
                    for j in (1, 2, 3)
                ]
                for i in (1, 2, 3)
            ]
        )
        translation_vector = np.array(
            [struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)]
        )
        transformation_dict[id] = AffineTransformation(
            np.zeros(3), rotation_matrix, translation_vector
        )
    return transformation_dict


def _parse_operation_expression(expression):
    """
    Get successive operation steps (IDs) for the given
    ``oper_expression``.
    Form the cartesian product, if necessary.
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    # example: '(X0)(1-10,21-25)' from 1a34
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for one_step_expr in expressions_per_step:
        one_step_op_ids = []
        for expr in one_step_expr.split(","):
            if "-" in expr:
                # Range of operation IDs, they must be integers
                first, last = expr.split("-")
                one_step_op_ids.extend(
                    [str(id) for id in range(int(first), int(last) + 1)]
                )
            else:
                # Single operation ID
                one_step_op_ids.append(expr)
        operations.append(one_step_op_ids)

    # Cartesian product of operations
    return list(itertools.product(*operations))


def _convert_string_to_sequence(string, stype):
    """
    Convert strings to `ProteinSequence` if `stype` is contained in
    ``proteinseq_type_list`` or to ``NucleotideSequence`` if `stype` is
    contained in ``_nucleotideseq_type_list``.
    """
    # sequence may be stored as multiline string
    string = string.replace("\n", "")
    if stype in _proteinseq_type_list:
        return ProteinSequence(string)
    elif stype in _nucleotideseq_type_list:
        string = string.replace("U", "T")
        return NucleotideSequence(string)
    elif stype in _other_type_list:
        return None
    else:
        raise InvalidFileError("mmCIF _entity_poly.type unsupported type: " + stype)


def get_unit_cell(
    pdbx_file,
    center=True,
    model=None,
    data_block=None,
    altloc="first",
    extra_fields=None,
    use_author_fields=True,
    include_bonds=False,
):
    """
    Build a structure model containing all symmetric copies of the structure within a
    single unit cell.

    This function receives the data from the ``symmetry`` and ``atom_site`` categories
    in the file.
    Consequently, these categories must be present in the file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
    center : bool, optional
        If set to true, each symmetric copy will be moved inside the unit cell
        dimensions, if its centroid is outside.
        By default, the copies are are created using the raw space group
        transformations, which may put them one unit cell length further away.
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
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
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
        to other categories in the file, the ``auth_xxx``
        fields are set by the author(s) of the structure and are
        consistent with the corresponding values in PDB files.
        If `use_author_fields` is true, the annotation arrays will be
        read from the ``auth_xxx`` fields (if applicable),
        otherwise from the the ``label_xxx`` fields.
    include_bonds : bool, optional
        If set to true, a :class:`BondList` will be created for the
        resulting :class:`AtomArray` containing the bond information
        from the file.
        Bonds, whose order could not be determined from the
        *Chemical Component Dictionary*
        (e.g. especially inter-residue bonds),
        have :attr:`BondType.ANY`, since the PDB format itself does
        not support bond orders.

    Returns
    -------
    unit_cell : AtomArray or AtomArrayStack
        The structure representing the unit cell.
        The return type depends on the `model` parameter.
        Contains the `sym_id` annotation, which enumerates the copies of the asymmetric
        unit in the unit cell.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1f2n.cif"))
    >>> unit_cell = get_unit_cell(file, model=1)
    """
    block = _get_block(pdbx_file, data_block)

    try:
        space_group = block["symmetry"]["space_group_name_H-M"].as_item()
    except KeyError:
        raise InvalidFileError("File has no 'symmetry.space_group_name_H-M' field")
    transforms = space_group_transforms(space_group)

    asym = get_structure(
        pdbx_file,
        model,
        data_block,
        altloc,
        extra_fields,
        use_author_fields,
        include_bonds,
    )

    fractional_asym_coord = coord_to_fraction(asym.coord, asym.box)
    unit_cell_copies = []
    for transform in transforms:
        fractional_coord = transform.apply(fractional_asym_coord)
        if center:
            # If the centroid is outside the box, move the copy inside the box
            orig_centroid = centroid(fractional_coord)
            new_centroid = orig_centroid % 1
            fractional_coord += (new_centroid - orig_centroid)[..., np.newaxis, :]
        unit_cell_copies.append(fraction_to_coord(fractional_coord, asym.box))

    unit_cell = repeat(asym, np.stack(unit_cell_copies, axis=0))
    unit_cell.set_annotation(
        "sym_id", np.repeat(np.arange(len(transforms)), asym.array_length())
    )
    return unit_cell


def get_sse(pdbx_file, data_block=None, match_model=None):
    """
    Get the secondary structure from a PDBx file.

    Parameters
    ----------
    pdbx_file : CIFFile or CIFBlock or BinaryCIFFile or BinaryCIFBlock
        The file object.
        The following categories are required:

        - ``entity_poly``
        - ``struct_conf`` (if alpha-helices are present)
        - ``struct_sheet_range`` (if beta-strands are present)
        - ``atom_site`` (if `match_model` is set)

    data_block : str, optional
        The name of the data block.
        Default is the first (and most times only) data block of the
        file.
        If the data block object is passed directly to `pdbx_file`,
        this parameter is ignored.
    match_model : None, optional
        If a model number is given, only secondary structure elements for residues are
        kept, that are resolved in the given model.
        This means secondary structure elements for residues that would not appear
        in a corresponding :class:`AtomArray` from :func:`get_structure()` are removed.
        By default, all residues in the sequence are kept.

    Returns
    -------
    sse_dict : dict of str -> ndarray, dtype=str
        The dictionary maps the chain ID (derived from ``auth_asym_id``) to the
        secondary structure of the respective chain.

        - ``"a"``: alpha-helix
        - ``"b"``: beta-strand
        - ``"c"``: coil or not an amino acid

        Each secondary structure element corresponds to the ``label_seq_id`` of the
        ``atom_site`` category.
        This means that the 0-th position of the array corresponds to the residue
        in ``atom_site`` with ``label_seq_id`` ``1``.

    Examples
    --------

    >>> import os.path
    >>> file = CIFFile.read(os.path.join(path_to_structures, "1aki.cif"))
    >>> sse = get_sse(file, match_model=1)
    >>> print(sse)
    {'A': array(['c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
                 'a', 'c', 'c', 'c', 'c', 'c', 'a', 'a', 'a', 'c', 'c', 'a', 'a',
                 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'c',
                 'c', 'c', 'c', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'b', 'b',
                 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
                 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
                 'c', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'a', 'a', 'a',
                 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'a',
                 'a', 'a', 'a', 'c', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'c',
                 'c', 'c', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c'],
                 dtype='<U1')}

    If only secondary structure elements for resolved residues are requested, the length
    of the returned array matches the number of peptide residues in the structure.

    >>> file = CIFFile.read(os.path.join(path_to_structures, "3o5r.cif"))
    >>> print(len(get_sse(file, match_model=1)["A"]))
    128
    >>> atoms = get_structure(file, model=1)
    >>> atoms = atoms[filter_amino_acids(atoms) & (atoms.chain_id == "A")]
    >>> print(get_residue_count(atoms))
    128
    """
    block = _get_block(pdbx_file, data_block)

    # Init all chains with "c" for coil
    sse_dict = {
        chain_id: np.repeat("c", len(sequence))
        for chain_id, sequence in get_sequence(block).items()
    }

    # Populate SSE arrays with helices and strands
    for sse_symbol, category_name in [
        ("a", "struct_conf"),
        ("b", "struct_sheet_range"),
    ]:
        if category_name in block:
            category = block[category_name]
            chains = category["beg_auth_asym_id"].as_array(str)
            start_positions = category["beg_label_seq_id"].as_array(int)
            end_positions = category["end_label_seq_id"].as_array(int)

            # set alpha helix positions
            for chain, start, end in zip(chains, start_positions, end_positions):
                # Translate the 1-based positions from PDBx into 0-based array indices
                sse_dict[chain][start - 1 : end] = sse_symbol

    if match_model is not None:
        model_atom_site = _filter_model(block["atom_site"], match_model)
        chain_ids = model_atom_site["auth_asym_id"].as_array(str)
        res_ids = model_atom_site["label_seq_id"].as_array(int, masked_value=-1)
        # Filter out masked residues, i.e. residues not part of a chain
        mask = res_ids != -1
        chain_ids = chain_ids[mask]
        res_ids = res_ids[mask]
        for chain_id, sse in sse_dict.items():
            res_ids_in_chain = res_ids[chain_ids == chain_id]
            # Transform from 1-based residue ID to 0-based index
            indices = np.unique(res_ids_in_chain) - 1
            sse_dict[chain_id] = sse[indices]

    return sse_dict
