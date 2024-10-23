# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["get_ccd", "get_from_ccd"]

import functools
from pathlib import Path
import numpy as np

CCD_DIR = Path(__file__).parent / "ccd"
SPECIAL_ID_COLUMN_NAMES = {
    "chem_comp": "id",
}
DEFAULT_ID_COLUMN_NAME = "comp_id"


@functools.cache
def get_ccd():
    """
    Get the internal subset of the PDB
    *Chemical Component Dictionary* (CCD).
    :footcite:`Westbrook2015`

    Returns
    -------
    ccd : BinaryCIFBlock
        The CCD.

    Warnings
    --------

    Consider the return value as read-only.
    As other functions cache data from it, changing data may lead to undefined
    behavior.

    References
    ----------

    .. footbibliography::

    """
    # Avoid circular import
    from biotite.structure.io.pdbx.bcif import BinaryCIFFile

    return BinaryCIFFile.read(CCD_DIR / "components.bcif").block


@functools.cache
def get_from_ccd(category_name, comp_id, column_name=None):
    """
    Get the rows for the given residue in the given category from the
    internal subset of the PDB *Chemical Component Dictionary* (CCD).
    :footcite:`Westbrook2015`

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

    Notes
    -----
    The returned values are cached for faster access in subsequent calls.

    References
    ----------

    .. footbibliography::

    """
    try:
        start, stop = _residue_index(category_name)[comp_id]
    except KeyError:
        return None

    category = get_ccd()[category_name]
    if column_name is None:
        return {
            col_name: category[col_name].as_array()[start:stop]
            for col_name in category.keys()
        }
    else:
        return category[column_name].as_array()[start:stop]


@functools.cache
def _residue_index(category_name):
    """
    Get the start and stop index for each component name in the given
    CCD category.

    Parameters
    ----------
    category_name : str
        The category to determine start and stop indices for each component in.

    Returns
    -------
    index : dict (str -> (int, int))
        The index maps each present component name to the corresponding
        start and exclusive stop index in `id_column`.
    """
    category = get_ccd()[category_name]
    id_column_name = SPECIAL_ID_COLUMN_NAMES.get(category_name, DEFAULT_ID_COLUMN_NAME)
    id_column = category[id_column_name].as_array()

    residue_starts = np.where(id_column[:-1] != id_column[1:])[0] + 1
    # The final start is the exclusive stop of last residue
    residue_starts = np.concatenate(([0], residue_starts, [len(id_column)]))
    index = {}
    for i in range(len(residue_starts) - 1):
        comp_id = id_column[residue_starts[i]].item()
        index[comp_id] = (residue_starts[i], residue_starts[i + 1])
    return index
