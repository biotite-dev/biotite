import gzip
import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path
import numpy as np
import requests
from biotite.structure.io.pdbx import *

TARGET_DIR = Path(__file__).parent / "src" / "biotite" / "structure" / "info" / "ccd"
CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
COMPONENT_GROUPS = {
    "amino_acids": [
        "D-beta-peptide, C-gamma linking",
        "D-gamma-peptide, C-delta linking",
        "D-peptide COOH carboxy terminus",
        "D-peptide NH3 amino terminus",
        "D-peptide linking",
        "L-beta-peptide, C-gamma linking",
        "L-gamma-peptide, C-delta linking",
        "L-peptide COOH carboxy terminus",
        "L-peptide NH3 amino terminus",
        "L-peptide linking",
        "peptide linking",
    ],
    "nucleotides": [
        "DNA OH 3 prime terminus",
        "DNA OH 5 prime terminus",
        "DNA linking",
        "L-DNA linking",
        "L-RNA linking",
        "RNA OH 3 prime terminus",
        "RNA OH 5 prime terminus",
        "RNA linking",
    ],
    "carbohydrates": [
        "D-saccharide",
        "D-saccharide, alpha linking",
        "D-saccharide, beta linking",
        "L-saccharide",
        "L-saccharide, alpha linking",
        "L-saccharide, beta linking",
        "saccharide",
    ],
}


def concatenate_ccd(categories=None):
    """
    Create the CCD in BinaryCIF format with each category contains the
    data of all blocks.

    Parameters
    ----------
    categories : list of str, optional
        The names of the categories to include.
        By default, all categories from the CCD are included.

    Returns
    -------
    compressed_file : BinaryCIFFile
        The compressed CCD in BinaryCIF format.
    """

    logging.info("Download and read CCD...")
    ccd_cif_text = gzip.decompress(requests.get(CCD_URL).content).decode()
    ccd_file = CIFFile.read(StringIO(ccd_cif_text))

    compressed_block = BinaryCIFBlock()
    if categories is None:
        categories = _list_all_category_names(ccd_file)
    for category_name in categories:
        logging.info(f"Concatenate and compress '{category_name}' category...")
        compressed_block[category_name] = compress(
            _concatenate_blocks_into_category(ccd_file, category_name)
        )

    logging.info("Write concatenated CCD into BinaryCIF...")
    compressed_file = BinaryCIFFile()
    compressed_file["components"] = compressed_block
    return compressed_file


def group_components(ccd, match_types):
    """
    Identify component IDs that matches a *given component type* from the given file.

    Parameters
    ----------
    ccd : BinaryCIFFile
        The file to look into-
    match_types : list of str
        The component types to extract.

    Returns
    -------
    comp_ids : list of str
        The extracted component IDs.
    """
    category = ccd.block["chem_comp"]
    comp_ids = category["id"].as_array()
    types = category["type"].as_array()
    # Ignore case
    return comp_ids[np.isin(np.char.lower(types), np.char.lower(match_types))].tolist()


def _concatenate_blocks_into_category(pdbx_file, category_name):
    """
    Concatenate the given category from all blocks into a single
    category.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The PDBx file, whose blocks should be concatenated.
    category_name : str
        The name of the category to concatenate.

    Returns
    -------
    category : BinaryCIFCategory
        The concatenated category.
    """
    columns_names = _list_all_column_names(pdbx_file, category_name)
    data_chunks = defaultdict(list)
    mask_chunks = defaultdict(list)
    for block in pdbx_file.values():
        if category_name not in block:
            continue
        category = block[category_name]
        for column_name in columns_names:
            if column_name in category:
                column = category[column_name]
                data_chunks[column_name].append(column.data.array)
                if column.mask is not None:
                    mask_chunks[column_name].append(column.mask.array)
                else:
                    mask_chunks[column_name].append(
                        np.full(category.row_count, MaskValue.PRESENT, dtype=np.uint8)
                    )
            else:
                # Column is missing in this block
                # -> handle it as data masked as 'missing'
                data_chunks[column_name].append(
                    # For now all arrays are of type string anyway,
                    # as they are read from a CIF file
                    np.full(category.row_count, "", dtype="U1")
                )
                mask_chunks[column_name].append(
                    np.full(category.row_count, MaskValue.MISSING, dtype=np.uint8)
                )

    bcif_columns = {}
    for col_name in columns_names:
        data = np.concatenate(data_chunks[col_name])
        mask = np.concatenate(mask_chunks[col_name])
        data = _into_fitting_type(data, mask)
        if np.all(mask == MaskValue.PRESENT):
            mask = None
        bcif_columns[col_name] = BinaryCIFColumn(data, mask)
    return BinaryCIFCategory(bcif_columns)


def _list_all_column_names(pdbx_file, category_name):
    """
    Get all columns that exist in any block for a given category.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The PDBx file to search in for the columns.
    category_name : str
        The name of the category to search in.

    Returns
    -------
    columns_names : list of str
        The names of the columns.
    """
    columns_names = set()
    for block in pdbx_file.values():
        if category_name in block:
            columns_names.update(block[category_name].keys())
    return sorted(columns_names)


def _list_all_category_names(pdbx_file):
    """
    Get all categories that exist in any block.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The PDBx file to search in for the columns.

    Returns
    -------
    columns_names : list of str
        The names of the columns.
    """
    category_names = set()
    for block in pdbx_file.values():
        category_names.update(block.keys())
    return sorted(category_names)


def _into_fitting_type(string_array, mask):
    """
    Try to find a numeric type for a string ndarray, if possible.

    Parameters
    ----------
    string_array : ndarray, dtype=string
        The array to convert.
    mask : ndarray, dtype=uint8
        Only values in `string_array` where the mask is ``MaskValue.PRESENT`` are
        considered for type conversion.

    Returns
    -------
    array : ndarray
        The array converted into an appropriate dtype.
    """
    mask = mask == MaskValue.PRESENT
    # Only try to find an appropriate dtype for unmasked values
    values = string_array[mask]
    try:
        # Try to fit into integer type
        values = values.astype(int)
    except ValueError:
        try:
            # Try to fit into float type
            values = values.astype(float)
        except ValueError:
            # Keep string type
            pass
    array = np.zeros(string_array.shape, dtype=values.dtype)
    array[mask] = values
    return array


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    compressed_ccd = concatenate_ccd(["chem_comp", "chem_comp_atom", "chem_comp_bond"])
    compressed_ccd.write(TARGET_DIR / "components.bcif")

    for super_group, groups in COMPONENT_GROUPS.items():
        logging.info(f"Identify all components belonging to '{super_group}' group...")
        components = group_components(compressed_ccd, groups)
        with open(TARGET_DIR / f"{super_group}.txt", "w") as file:
            file.write("\n".join(components) + "\n")
