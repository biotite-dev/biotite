import gzip
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import numpy as np
import requests
from biotite.structure.io.pdbx import *


class ComponentError(Exception):
    pass


@dataclass
class ColumnInfo:
    """
    Defines how to re-econde a column.

    Attributes
    ----------
    dtype : dtype
        The data type of the column.
    fill_value : object
        The value to fill masked values with.
    encoding : list of Encoding
        The encodings to apply to the data.
    alternative : str, optional
        The name of an alternative column to use, if the original column
        contains masked values and no `fill_value` is given.
    """

    dtype: ...
    encoding: ...
    fill_value: ... = None
    alternative: ... = None


MAIN_COLUMNS = {
    "id": ColumnInfo(
        "U5",
        [
            StringArrayEncoding(
                data_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=2, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "name": ColumnInfo(
        str,
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT32)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "type": ColumnInfo(
        str,
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "formula_weight": ColumnInfo(
        "f8",
        [
            FixedPointEncoding(factor=1000, src_type=TypeCode.FLOAT64),
            ByteArrayEncoding(),
        ],
        fill_value=0,
    ),
    "one_letter_code": ColumnInfo(
        "U1",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
        fill_value="",
    ),
}


ATOM_COLUMNS = {
    "comp_id": ColumnInfo(
        "U5",
        [
            StringArrayEncoding(
                data_encoding=[
                    RunLengthEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=2, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "atom_id": ColumnInfo(
        "U6",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "alt_atom_id": ColumnInfo(
        "U6",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "type_symbol": ColumnInfo(
        "U2",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT8)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "charge": ColumnInfo("i1", [ByteArrayEncoding(type=TypeCode.INT8)], fill_value=0),
    "pdbx_model_Cartn_x_ideal": ColumnInfo(
        "f4",
        [
            FixedPointEncoding(factor=100),
            IntegerPackingEncoding(byte_count=2, is_unsigned=False),
            ByteArrayEncoding(),
        ],
        alternative="model_Cartn_x",
    ),
    "pdbx_model_Cartn_y_ideal": ColumnInfo(
        "f4",
        [
            FixedPointEncoding(factor=100),
            IntegerPackingEncoding(byte_count=2, is_unsigned=False),
            ByteArrayEncoding(),
        ],
        alternative="model_Cartn_y",
    ),
    "pdbx_model_Cartn_z_ideal": ColumnInfo(
        "f4",
        [
            FixedPointEncoding(factor=100),
            IntegerPackingEncoding(byte_count=2, is_unsigned=False),
            ByteArrayEncoding(),
        ],
        alternative="model_Cartn_z",
    ),
}

BOND_COLUMNS = {
    "comp_id": ColumnInfo(
        "U5",
        [
            StringArrayEncoding(
                data_encoding=[
                    RunLengthEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=2, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "atom_id_1": ColumnInfo(
        "U6",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "atom_id_2": ColumnInfo(
        "U6",
        [
            StringArrayEncoding(
                # The unique strings in the column are sorted
                # -> Indices do not follow distinct pattern
                data_encoding=[ByteArrayEncoding(type=TypeCode.INT16)],
                offset_encoding=[
                    DeltaEncoding(src_type=TypeCode.INT32),
                    RunLengthEncoding(),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
            )
        ],
    ),
    "value_order": ColumnInfo(
        "U4",
        [
            StringArrayEncoding(
                data_encoding=[
                    RunLengthEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
                offset_encoding=[ByteArrayEncoding(type=TypeCode.UINT8)],
            )
        ],
    ),
    "pdbx_aromatic_flag": ColumnInfo(
        "U1",
        [
            StringArrayEncoding(
                data_encoding=[
                    RunLengthEncoding(src_type=TypeCode.INT32),
                    IntegerPackingEncoding(byte_count=1, is_unsigned=True),
                    ByteArrayEncoding(),
                ],
                offset_encoding=[ByteArrayEncoding(type=TypeCode.UINT8)],
            )
        ],
    ),
}


CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"


def check_presence(pdbx_file, category_name, column_names):
    """
    For each block in the file, check if each of the given column names
    are present and unmasked.
    Alternatively, all given column names may be masked/missing.

    This is used to ensure that coordinates are consistent:
    If one dimension would be missing and another one would not,
    the fallback of only one dimension would be used.
    In consequence, the molecule coordinates would be distorted.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The file to check.
    category_name : str
        The name of the category to check.
    column_names : list of str
        The names of the columns to check.
    """
    for _, block in pdbx_file.items():
        if category_name not in block:
            continue
        category = block[category_name]

        is_present = column_names[0] in category
        for name in column_names:
            if (name in category) != is_present:
                raise ComponentError("Only some column names are missing")
        if not is_present:
            return

        is_unmasked = category[column_names[0]].mask is None
        for name in column_names:
            if (category[name].mask is None) != is_unmasked:
                raise ComponentError("Only some column names are masked")


def concatenate_blocks_into_category(pdbx_file, category_name, column_infos):
    """
    Concatenate the given category from all blocks into a single
    category.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The PDBx file, whose blocks should be concatenated.
    category_name : str
        The name of the category to concatenate.
    column_infos : dict (str -> ColumnInfo)
        Defines which columns of the category to keep and how to re-encode
        them, where keys are the column names.

    Returns
    -------
    category : BinaryCIFCategory
        The concatenated category.
    """
    column_chunks = {col_name: [] for col_name in column_infos.keys()}
    for comp_id, block in pdbx_file.items():
        try:
            if category_name not in block:
                raise ComponentError(f"Block has no category '{category_name}'")
            chunk = {}
            category = block[category_name]
            for col_name, info in column_infos.items():
                col = category.get(col_name)
                if col is None or (col.mask is not None and info.fill_value is None):
                    # Some/all values are missing and there is no default
                    # -> Try alternative
                    if info.alternative is not None:
                        col = category[info.alternative]
                        if col.mask is not None:
                            raise ComponentError(
                                f"Missing values in alternative "
                                f"'{info.alternative}'"
                            )
                    else:
                        raise ComponentError(f"Missing values in column '{col_name}'")
                data_array = col.as_array(info.dtype, info.fill_value)
                chunk[col_name] = data_array
        except ComponentError as e:
            logging.warning(f"Skipping '{comp_id}': {e}")
        # Append all columns in the chunk after the try-except block
        # to avoid appending incomplete chunks
        else:
            for col_name, data_array in chunk.items():
                column_chunks[col_name].append(data_array)
    return BinaryCIFCategory(
        {
            col_name: BinaryCIFData(
                array=np.concatenate(col_data), encoding=column_infos[col_name].encoding
            )
            for col_name, col_data in column_chunks.items()
        }
    )


def extract_component_groups(type_dict, include, exclude, file_name):
    """
    Extract component IDs that matches a given group from the given
    dictionary.

    Parameters
    ----------
    type_dict : dict
        A dictionary that maps component IDs to their type.
    include, exclude : list of str
        The keywords to be matched.
    file_name : Path
        The path the output file to write the extracted component IDs
        to.
    """
    # Find components that matches the given keywords
    comp_ids_for_group = []
    types_for_group = set()
    for comp_id, comp_type in type_dict.items():
        if any(keyword in comp_type.lower() for keyword in exclude):
            # 'xxx-like' components are not considered
            # as they are not real 'xxx'
            continue
        if any(keyword in comp_type.lower() for keyword in include):
            comp_ids_for_group.append(comp_id)
            types_for_group.add(comp_type.lower())
    # Remove extracted components from dict
    for comp_id in comp_ids_for_group:
        del type_dict[comp_id]
    # Write extracted components into output file
    logging.info(
        f"Using the following types for '{file_name.name}':\n"
        + ", ".join(types_for_group)
    )
    with open(file_name, "w") as file:
        for comp_id in comp_ids_for_group:
            file.write(comp_id + "\n")


def setup_ccd(target_diriectory):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    target_diriectory.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading and reading CCD...")
    ccd_cif_text = gzip.decompress(requests.get(CCD_URL).content).decode()
    ccd_file = CIFFile.read(StringIO(ccd_cif_text))

    logging.info("Checking for consistent coordinates...")
    check_presence(
        ccd_file, "chem_comp_atom", ["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"]
    )
    check_presence(
        ccd_file,
        "chem_comp_atom",
        ["model_Cartn_x_ideal", "model_Cartn_y_ideal", "model_Cartn_z_ideal"],
    )

    logging.info("Extracting component groups...")
    type_dict = {
        comp_id: block["chem_comp"]["type"].as_item()
        for comp_id, block in ccd_file.items()
    }
    extract_component_groups(
        type_dict,
        ["peptide", "amino"],
        ["peptide-like"],
        target_diriectory / "amino_acids.txt",
    )
    extract_component_groups(
        type_dict, ["rna", "dna"], [], target_diriectory / "nucleotides.txt"
    )
    extract_component_groups(
        type_dict, ["saccharide"], [], target_diriectory / "carbohydrates.txt"
    )
    remaining_types = set(type_dict.values())
    logging.info(
        "The following types are not used in any group:\n" + ", ".join(remaining_types)
    )

    compressed_block = BinaryCIFBlock()
    for category_name, column_infos in [
        ("chem_comp", MAIN_COLUMNS),
        ("chem_comp_atom", ATOM_COLUMNS),
        ("chem_comp_bond", BOND_COLUMNS),
    ]:
        logging.info(f"Concatenate '{category_name}' category...")
        compressed_block[category_name] = concatenate_blocks_into_category(
            ccd_file, category_name, column_infos
        )

    logging.info("Write concatenated CCD into BinaryCIF...")
    compressed_file = BinaryCIFFile()
    compressed_file["components"] = compressed_block
    compressed_file.write(target_diriectory / "components.bcif")


setup_ccd(Path(__file__).parent / "src" / "biotite" / "structure" / "info" / "ccd")
