# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite"
__author__ = "Patrick Kunzmann"
__all__ = ["map_unique"]

from collections.abc import Callable
from typing import Any, overload
import numpy as np
from biotite.typing import K, N, NDArray1, NDArray2


@overload
def map_unique(
    function: Callable[..., tuple[Any, ...]], *arrays: NDArray1[N, Any]
) -> NDArray2[N, K, Any]: ...
@overload
def map_unique(
    function: Callable[..., Any], *arrays: NDArray1[N, Any]
) -> NDArray1[N, Any]: ...
def map_unique(
    function: Callable[..., Any], *arrays: NDArray1[N, Any]
) -> NDArray1[N, Any] | NDArray2[N, K, Any]:
    """
    Apply a function to each element of the given arrays, but call it only
    once per unique element.

    This is much faster than calling `function` for each element, if the
    arrays contain only few unique elements, which is usually the case for
    annotation arrays of an :class:`AtomArray`.

    Parameters
    ----------
    function : Callable
        The function to apply.
        It is called with one positional argument per given array, i.e. the
        respective element of each array as Python object.
        It must return either a scalar value or a tuple.
    *arrays : ndarray, shape=(n,)
        The arrays to apply the function to.
        If multiple arrays are given, `function` is called once per unique
        combination of elements.
        All arrays must have the same length.

    Returns
    -------
    mapped : ndarray, shape=(n,) or shape=(n,k)
        The return value of `function` for each element.
        If `function` returns tuples of length *k*, the array has a second
        dimension for the tuple elements.

    Examples
    --------

    >>> elements = np.array(["C", "N", "C", "O", "N"])
    >>> print(map_unique(str.lower, elements))
    ['c' 'n' 'c' 'o' 'n']
    >>> atom_names = np.array(["CA", "N", "CB", "O", "ND2"])
    >>> print(map_unique(lambda name, elem: (name + "_" + elem, len(name)), atom_names, elements))
    [['CA_C' '2']
     ['N_N' '1']
     ['CB_C' '2']
     ['O_O' '1']
     ['ND2_N' '3']]
    """
    if len(arrays) == 0:
        raise TypeError("At least one array is required")
    length = len(arrays[0])
    if any(len(array) != length for array in arrays):
        raise IndexError("All arrays must have the same length")

    if len(arrays) == 1:
        keys = arrays[0]
    else:
        # Encode the elements of each array as integers, so that each
        # combination of elements is represented by a single integer key
        keys = np.zeros(length, dtype=np.int64)
        for array in arrays:
            unique_elements, codes = np.unique_inverse(array)
            keys = keys * len(unique_elements) + codes
    unique = np.unique_all(keys)
    # The first occurrence of each unique key is representative for the key
    unique_results = [
        function(*elements)
        for elements in zip(*(array[unique.indices].tolist() for array in arrays))
    ]
    return np.array(unique_results)[unique.inverse_indices]
