# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility for handling data on chain level, rather than
atom level.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "get_chain_starts",
    "apply_chain_wise",
    "spread_chain_wise",
    "get_chain_masks",
    "get_chain_starts_for",
    "get_chain_positions",
    "chain_iter",
    "get_chains",
    "get_chain_count",
    "chain_iter",
]

from biotite.structure.segments import (
    apply_segment_wise,
    get_segment_masks,
    get_segment_positions,
    get_segment_starts,
    get_segment_starts_for,
    segment_iter,
    spread_segment_wise,
)


def get_chain_starts(array, add_exclusive_stop=False, extra_categories=()):
    """
    Get the indices in an atom array, which indicates the beginning of
    a new chain.

    A new chain starts, when the chain or sym ID changes or when the residue ID
    decreases.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to get the chain starts from.
    add_exclusive_stop : bool, optional
        If true, the exclusive stop of the input atom array, i.e.
        ``array.array_length()``, is added to the returned array of
        start indices as last element.
    extra_categories : tuple of str, optional
        Additional annotation categories that induce the start of a new chain,
        when their value change from one atom to the next.

    Returns
    -------
    starts : ndarray, dtype=int
        The start indices of new chains in `array`.

    Notes
    -----
    This method is internally used by all other chain-related
    functions.
    """
    categories = ["chain_id"] + list(extra_categories)
    if "sym_id" in array.get_annotation_categories():
        categories.append("sym_id")
    return get_segment_starts(
        array,
        add_exclusive_stop,
        continuous_categories=("res_id",),
        equal_categories=categories,
    )


def apply_chain_wise(array, data, function, axis=None):
    """
    Apply a function to intervals of data, where each interval
    corresponds to one chain.

    The function takes an atom array (stack) and an data array
    (`ndarray`) of the same length. The function iterates through the
    chain IDs of the atom array (stack) and identifies intervals of
    the same ID. Then the data is
    partitioned into the same intervals, and each interval (also an
    :class:`ndarray`) is put as parameter into `function`. Each return value is
    stored as element in the resulting :class:`ndarray`, therefore each element
    corresponds to one chain.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the chains from.
    data : ndarray
        The data, whose intervals are the parameter for `function`. Must
        have same length as `array`.
    function : function
        The `function` must have either the form *f(data)* or
        *f(data, axis)* in case `axis` is given. Every `function` call
        must return a value with the same shape and data type.
    axis : int, optional
        This value is given to the `axis` parameter of `function`.

    Returns
    -------
    processed_data : ndarray
        Chain-wise evaluation of `data` by `function`. The size of the
        first dimension of this array is equal to the amount of
        chains.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    return apply_segment_wise(starts, data, function, axis)


def spread_chain_wise(array, input_data):
    """
    Expand chain-wise data to atom-wise data.

    Each value in the chain-wise input is assigned to all atoms of
    this chain:

    ``output_data[i] = input_data[j]``,
    *i* is incremented from atom to atom,
    *j* is incremented every chain change.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the chains from.
    input_data : ndarray
        The data to be spread. The length of axis=0 must be equal to
        the amount of different chain IDs in `array`.

    Returns
    -------
    output_data : ndarray
        Chain-wise spread `input_data`. Length is the same as
        `array_length()` of `array`.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    return spread_segment_wise(starts, input_data)


def get_chain_masks(array, indices):
    """
    Get boolean masks indicating the chains to which the given atom
    indices belong.

    Parameters
    ----------
    array : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The atom array (stack) to determine the chains from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices indicate the atoms to get the corresponding
        chains for.
        Negative indices are not allowed.

    Returns
    -------
    chains_masks : ndarray, dtype=bool, shape=(k,n)
        Multiple boolean masks, one for each given index in `indices`.
        Each array masks the atoms that belong to the same chain as
        the atom at the given index.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    return get_segment_masks(starts, indices)


def get_chain_starts_for(array, indices):
    """
    For each given atom index, get the index that points to the
    start of the chain that atom belongs to.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the chains from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices point to the atoms to get the corresponding
        chain starts for.
        Negative indices are not allowed.

    Returns
    -------
    start_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the chain starts for the input
        `indices`.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    return get_segment_starts_for(starts, indices)


def get_chain_positions(array, indices):
    """
    For each given atom index, obtain the position of the chain
    corresponding to this index in the input `array`.

    For example, the position of the first chain in the atom array is
    ``0``, the the position of the second chain is ``1``, etc.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the chains from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices point to the atoms to get the corresponding
        chain positions for.
        Negative indices are not allowed.

    Returns
    -------
    start_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the position of the chains.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    return get_segment_positions(starts, indices)


def get_chains(array):
    """
    Get the chain IDs of an atom array (stack).

    The chains are listed in the same order they occur in the array
    (stack).

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack), where the chains are determined.

    Returns
    -------
    ids : ndarray, dtype=str
        List of chain IDs.
    """
    return array.chain_id[get_chain_starts(array)]


def get_chain_count(array):
    """
    Get the amount of chains in an atom array (stack).

    The count is determined from the `chain_id` annotation.
    Each time the chain ID changes, the count is incremented.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack), where the chains are counted.

    Returns
    -------
    count : int
        Amount of chains.
    """
    return len(get_chain_starts(array))


def chain_iter(array):
    """
    Iterate over all chains in an atom array (stack).

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to iterate over.

    Yields
    ------
    chain : AtomArray or AtomArrayStack
        A single chain of the input `array`.
    """
    starts = get_chain_starts(array, add_exclusive_stop=True)
    for chain in segment_iter(array, starts):
        yield chain
