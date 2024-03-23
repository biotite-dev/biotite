# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["get_ccd", "get_from_ccd"]

from pathlib import Path
import numpy as np


CCD_DIR = Path(__file__).parent / "ccd"
INDEX_COLUMN_NAME = {
    "chem_comp": "id",
    "chem_comp_atom": "comp_id",
    "chem_comp_bond": "comp_id",
}

_ccd_block = None
# For each category this index gives the start and stop for each residue
_residue_index = {}


def get_ccd():
    """
    Get the PDB *Chemical Component Dictionary* (CCD).

    Returns
    -------
    ccd : BinaryCIFFile
        The CCD.
    """
    # Avoid circular import
    from ..io.pdbx.bcif import BinaryCIFFile

    global _ccd_block
    if _ccd_block is None:
        # Load CCD once and cache it for subsequent calls
        _ccd_block = BinaryCIFFile.read(CCD_DIR / "components.bcif").block
    return _ccd_block


def get_from_ccd(category_name, comp_id, column_name=None):
    """
    Get the rows for the given residue in the given category from the
    PDB *Chemical Component Dictionary* (CCD).

    Parameters
    ----------
    category_name : str
        The category in the CCD.
    comp_id : str
        The residue identifier, i.e. the ``res_name``.
    column_name : str, optional
        The name of the column to be retrieved.
        If None, all columns are returned as dictionary.
        By default None.

    Returns
    -------
    value : ndarray or dict or None
        The array of the given column or all columns as dictionary.
        ``None`` if the `comp_id` is not found in the category.
    """
    global _residue_index
    ccd = get_ccd()
    category = ccd[category_name]
    if category_name not in _residue_index:
        _residue_index[category_name] = _index_residues(
            category[INDEX_COLUMN_NAME[category_name]].as_array()
        )
    try:
        start, stop = _residue_index[category_name][comp_id]
    except KeyError:
        return None

    if column_name is None:
        return {
            col_name: category[col_name].as_array()[start:stop]
            for col_name in category.keys()
        }
    else:
        return category[column_name].as_array()[start:stop]


def _index_residues(id_column):
    residue_starts = np.where(id_column[:-1] != id_column[1:])[0] + 1
    # The final start is the exclusive stop of last residue
    residue_starts = np.concatenate(([0], residue_starts, [len(id_column)]))
    index = {}
    for i in range(len(residue_starts)-1):
        comp_id = id_column[residue_starts[i]].item()
        index[comp_id] = (residue_starts[i], residue_starts[i+1])
    return index