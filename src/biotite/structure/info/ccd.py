# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["get_ccd", "set_ccd_path", "get_from_ccd"]

import functools
import importlib
import inspect
import pkgutil
from pathlib import Path
import numpy as np

_CCD_FILE = Path(__file__).parent / "components.bcif"
_SPECIAL_ID_COLUMN_NAMES = {
    "chem_comp": "id",
}
_DEFAULT_ID_COLUMN_NAME = "comp_id"


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
        It contains the categories `chem_comp`, `chem_comp_atom` and `chem_comp_bond`.

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

    try:
        return BinaryCIFFile.read(_CCD_FILE).block
    except FileNotFoundError:
        raise RuntimeError(
            "Internal CCD not found. Please run 'python -m biotite.setup_ccd'."
        )


def set_ccd_path(ccd_path):
    """
    Replace the internal *Chemical Component Dictionary* (CCD) with a custom one.

    This function also clears the cache of functions depending on the CCD to ensure
    that the new CCD is used.

    Parameters
    ----------
    ccd_path : path-like
        The path to the custom CCD in BinaryCIF format, prepared with the
        ``setup_ccd.py`` module.

    Notes
    -----
    This function is intended for advanced users who need to add information for
    compounds, which are not part of the internal CCD.
    The reason might be that an updated version already exists upstream or that
    the user wants to add custom compounds to the CCD.
    """
    global _CCD_FILE
    _CCD_FILE = Path(ccd_path)

    # Clear caches in all functions in biotite.structure.info
    info_modules = [
        importlib.import_module(f"biotite.structure.info.{mod_name}")
        for _, mod_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)])
    ]
    for module in info_modules:
        for _, function in inspect.getmembers(module, callable):
            if hasattr(function, "cache_clear"):
                function.cache_clear()


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
    slice : BinaryCIFCategory or BinaryCIFColumn
        The category or column (if `column_name` is provided) containing only the rows
        for the given residue.

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
        return _filter_category(category, slice(start, stop))
    else:
        return _filter_column(category[column_name], slice(start, stop))


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
    id_column_name = _SPECIAL_ID_COLUMN_NAMES.get(
        category_name, _DEFAULT_ID_COLUMN_NAME
    )
    id_column = category[id_column_name].as_array()

    residue_starts = np.where(id_column[:-1] != id_column[1:])[0] + 1
    # The final start is the exclusive stop of last residue
    residue_starts = np.concatenate(([0], residue_starts, [len(id_column)]))
    index = {}
    for i in range(len(residue_starts) - 1):
        comp_id = id_column[residue_starts[i]].item()
        index[comp_id] = (residue_starts[i], residue_starts[i + 1])
    return index


def _filter_category(category, index):
    """
    Reduce the category to the values for the given index.∂
    """
    # Avoid circular import
    from biotite.structure.io.pdbx.bcif import BinaryCIFCategory

    return BinaryCIFCategory(
        {key: _filter_column(column, index) for key, column in category.items()}
    )


def _filter_column(column, index):
    """
    Reduce the column to the values for the given index.
    """
    # Avoid circular import
    from biotite.structure.io.pdbx.bcif import BinaryCIFColumn, BinaryCIFData
    from biotite.structure.io.pdbx.component import MaskValue

    data_array = column.data.array[index]
    mask_array = column.mask.array[index] if column.mask is not None else None
    return BinaryCIFColumn(
        BinaryCIFData(data_array),
        (
            BinaryCIFData(mask_array)
            if column.mask is not None and (mask_array != MaskValue.PRESENT).any()
            else None
        ),
    )
