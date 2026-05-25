# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Common functions used by a number of subpackages.
"""

__name__ = "biotite.structure.io"
__author__ = "Patrick Kunzmann"
__all__ = [
    "number_of_integer_digits",
    "convert_unicode_to_uint32",
    "convert_uint32_to_unicode",
]

import numpy as np


def number_of_integer_digits(values):
    """
    Get the maximum number of characters needed to represent the
    pre-decimal positions of the given numeric values.

    Parameters
    ----------
    values : ndarray, dtype=float
        The values to be checked.

    Returns
    -------
    n_digits : int
        The maximum number of characters needed to represent the
        pre-decimal positions of the given numeric values.
    """
    if len(values) == 0:
        return 0
    values = values.astype(int, copy=False)
    n_digits = 0
    n_digits = max(n_digits, len(str(np.min(values))))
    n_digits = max(n_digits, len(str(np.max(values))))
    return n_digits


def convert_unicode_to_uint32(array):
    """
    Convert a unicode string array into a 2D uint32 array.

    The second dimension corresponds to the character position within a
    string.

    Parameters
    ----------
    array : ndarray, dtype=unicode
        The unicode string array to be converted.

    Returns
    -------
    array : ndarray, dtype=uint32
        The converted unicode string array.
        Each element along the second dimension corresponds to an UTF-32 symbol.
    """
    dtype = array.dtype
    if not np.issubdtype(dtype, np.str_):
        raise TypeError("Expected unicode string array")
    length = array.shape[0]
    n_char = dtype.itemsize // 4
    return np.frombuffer(array, dtype=np.uint32).reshape(length, n_char)


def convert_uint32_to_unicode(array):
    """
    Convert a 2D uint32 array into a unicode string array.

    This is the inverse of :func:`convert_unicode_to_uint32`.

    Parameters
    ----------
    array : ndarray, dtype=uint32, shape=(n, m)
        The uint32 array to be converted.
        Each element along the second dimension corresponds to an UTF-32 symbol.

    Returns
    -------
    array : ndarray, dtype=unicode
        The converted unicode string array.
    """
    if array.ndim != 2:
        raise ValueError("Expected 2D array")
    length, n_char = array.shape
    return np.frombuffer(
        array.astype(np.uint32, copy=False).tobytes(), dtype=f"<U{n_char}"
    )
