# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.sequence"
__author__ = "Patrick Kunzmann"
__all__ = ["find_subsequence", "find_symbol", "find_symbol_first",
           "find_symbol_last"]

import numpy as np


def find_subsequence(sequence, query):
    """
    Find a subsequence in a sequence.
    
    Parameters
    ----------
    sequence : Sequence
        The sequence to find the subsequence in.
    query : Sequence
        The potential subsequence. Its alphabet must extend the
        `sequence` alphabet.
    
    Returns
    -------
    match_indices : ndarray
        The starting indices in `sequence`, where `query` has been
        found. The array is empty if no match has been found.
    
    Raises
    ------
    ValueError
        If the `query` alphabet does not extend the `sequence` alphabet.
    
    Examples
    --------
    
    >>> main_seq = NucleotideSequence("ACTGAATGA")
    >>> sub_seq = NucleotideSequence("TGA")
    >>> print(find_subsequence(main_seq, sub_seq))
    [2 6]
    
    """
    if not sequence.get_alphabet().extends(query.get_alphabet()):
        raise ValueError("The sequences alphabets are not equal")
    match_indices = []
    frame_size = len(query)
    for i in range(len(sequence) - frame_size + 1):
        sub_seq_code = sequence.code[i : i + frame_size]
        if np.array_equal(query.code, sub_seq_code):
            match_indices.append(i)
    return np.array(match_indices)

def find_symbol(sequence, symbol):
    """
    Find a symbol in a sequence.
    
    Parameters
    ----------
    sequence : Sequence
        The sequence to find the symbol in.
    symbol : object
        The symbol to be found in `sequence`.
    
    Returns
    -------
    match_indices : ndarray
        The indices in `sequence`, where `symbol` has been found.
    """
    code = sequence.get_alphabet().encode(symbol)
    return np.where(sequence.code == code)[0]

def find_symbol_first(sequence, symbol):
    """
    Find first occurence of a symbol in a sequence.
    
    Parameters
    ----------
    sequence : Sequence
        The sequence to find the symbol in.
    symbol : object
        The symbol to be found in `sequence`.
    
    Returns
    -------
    first_index : int
        The first index of `symbol` in `sequence`. If `symbol` is not in
        `sequence`, -1 is returned.
    """
    match_i = find_symbol(sequence, symbol)
    if len(match_i) == 0:
        return -1
    return np.min(match_i)
    
def find_symbol_last(sequence, symbol):
    """
    Find last occurence of a symbol in a sequence.
    
    Parameters
    ----------
    sequence : Sequence
        The sequence to find the symbol in.
    symbol : object
        The symbol to be found in `sequence`.
    
    Returns
    -------
    flast_index : int
        The last index of `symbol` in `sequence`. If `symbol` is not in
        `sequence`, -1 is returned.
    """
    match_i = find_symbol(sequence, symbol)
    if len(match_i) == 0:
        return -1
    return np.max(match_i)
