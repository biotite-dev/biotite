# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility for handling data on residue level, rather than
atom level.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["get_residue_starts", "apply_residue_wise", "spread_residue_wise",
           "get_residues", "get_residue_count"]

import numpy as np
from .atoms import AtomArray, AtomArrayStack


def get_residue_starts(array):
    """
    Get the indices in an atom array, whcih indicates the beginning of
    a residue.
    
    A new residue starts, either when the residue ID or chain ID
    changes from one to the next atom.
    If the residue ID is *-1*, a residue name change, instead of a
    residue ID change, is interpreted as new residue.

    This method is internally used by all other residue-related
    functions.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to get the residue starts from.
        
    Returns
    -------
    starts : ndarray
        The start indices of resdiues in `array`.
    """
    chain_ids = array.chain_id
    res_ids = array.res_id
    res_names = array.res_name

    # Maximum length is length of atom array
    starts = np.zeros(array.array_length(), dtype=int)
    starts[0] = 0
    i = 1
    # Variables for current values
    curr_chain_id = chain_ids[0]
    curr_res_id = res_ids[0]
    curr_res_name = res_names[0]

    for j in range(array.array_length()):
        if res_ids[j] != -1:
            if curr_res_id != res_ids[j] or curr_chain_id != chain_ids[j]:
                starts[i] = j
                i += 1
                curr_chain_id = chain_ids[j]
                curr_res_id   = res_ids[j]
                curr_res_name = res_names[j]
        else:
            # Residue ID = -1 -> Hetero residue
            # -> Cannot rely on residue ID for residue distinction
            # -> Fall back on residue names
            if curr_res_name != res_names[j] or curr_chain_id != chain_ids[j]:
                starts[i] = j
                i += 1
                curr_chain_id = chain_ids[j]
                curr_res_id   = res_ids[j]
                curr_res_name = res_names[j]
    # Trim to correct size
    return starts[:i]


def apply_residue_wise(array, data, function, axis=None):
    """
    Apply a function to intervals of data, where each interval
    corresponds to one residue.
    
    The function takes an atom array (stack) and an data array
    (`ndarray`) of the same length. The function iterates through the
    residue IDs of the atom array (stack) and identifies intervals of
    the same ID. Then the data is
    partitioned into the same intervals, and each interval (also an
    `ndarray`) is put as parameter into `function`. Each return value is
    stored as element in the resulting `ndarray`, therefore each element
    corresponds to one residue. 
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The `res_id` annotation array is taken from `array` in order to
        determine the intervals.
    data : ndarray
        The data, whose intervals are the parameter for `function`. Must
        have same length as `array`.
    function : function
        The `function` must have either the form *f(data)* or
        *f(data, axis)* in case `axis` is given. Every `function` call
        must return a value with the same shape and data type.
        
    Returns
    -------
    processed_data : ndarray
        Residue-wise evaluation of `data` by `function`. The size of the
        first dimension of this array is equal to the amount of
        residues.
        
    Examples
    --------
    Calculate residue-wise SASA from atom-wise SASA of a 20 residue
    peptide.
    
    >>> sasa_per_atom = sasa(atom_array)
    >>> print(len(sasa_per_atom))
    304
    >>> sasa_per_residue = apply_residue_wise(atom_array, sasa_per_atom, np.nansum)
    >>> print(len(sasa_per_residue))
    20
    >>> print(sasa_per_residue)
    [ 173.71750737  114.52487523   99.08331902  111.95128253  115.81167158
       20.58874161   96.50972632  138.9740059    74.63418835   37.31709418
        0.          113.23807888  118.38526428   28.30951972   84.92855916
      110.66448618  110.66448618   72.06059565   45.03787228  189.15906358]

    Calculate the controids of each residue for the same peptide.
        
    >>> file = fetch("1l2y","cif",temp_dir())
    >>> atom_array = get_structure_from(file)[0]
    >>> print(len(atom_array))
    304
    >>> centroids = apply_residue_wise(atom_array, atom_array.coord,
    ...                                np.average, axis=0)
    >>> print(len(centroids))
    20
    >>> print(centroids)
    [[ -9.5819375    3.3778125   -2.0728125 ]
     [ -4.66952632   5.81573684  -1.85989474]
     [ -2.46080952   3.05966667   3.07604762]
     [ -7.21084211  -0.39636842   1.01315789]
     [ -4.69782353  -1.08047059  -4.28411765]
     [  1.172125     0.20641667   1.038375  ]
     [ -2.16005263  -2.24494737   3.54052632]
     [ -3.68231818  -5.53977273  -2.89527273]
     [  0.71108333  -5.40941667  -2.5495    ]
     [  2.00242857  -6.32171429   1.69528571]
     [  2.79857143  -3.14         2.32742857]
     [  5.90071429  -2.48892857   4.84457143]
     [  6.75372727  -6.71236364   3.09418182]
     [  5.69927273  -5.10063636  -1.20918182]
     [  9.29542857  -2.96957143  -1.83528571]
     [  5.51795833  -1.52125     -3.47266667]
     [  7.21892857   3.67321429  -0.68435714]
     [  4.00664286   4.364        2.67385714]
     [  0.34114286   5.57528571  -0.25428571]
     [  1.194       10.41625      1.13016667]]    
    """
    starts = np.append(get_residue_starts(array), [array.array_length()])
    # The result array
    processed_data = None
    for i in range(len(starts)-1):
        interval = data[starts[i]:starts[i+1]]
        if axis == None:
            value = function(interval)
        else:
            value = function(interval, axis=axis)
        value = function(interval, axis=axis)
        # Identify the shape of the resulting array by evaluation
        # of the function return value for the first interval
        if processed_data is None:
            if isinstance(value, np.ndarray):
                # Maximum length of the processed data
                # is length of interval of size 1 -> length of all IDs
                # (equal to atom array length)
                processed_data = np.zeros((len(starts)-1,) + value.shape,
                                          dtype=value.dtype)
            else:
                # Scalar value -> one dimensional result array
                processed_data = np.zeros(len(starts)-1,
                                          dtype=type(value))
        # Write values into result arrays
        processed_data[i] = value
    return processed_data


def spread_residue_wise(array, input_data):
    """
    Creates an `ndarray` with residue-wise spreaded values from an input
    `ndarray`.
    
    ``output_data[i] = input_data[j]``,
    *i* is incremented from atom to atom,
    *j* is incremented every residue change.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The data is spreaded over `array`'s 'res_id` annotation array.
    input_data : ndarray
        The data to be spreaded. The length of axis=0 must be equal to
        the amount of different residue IDs in `array`.
        
    Returns
    -------
    output_data : ndarray
        Residue-wise spreaded `input_data`. Length is the same as
        `array_length()` of `array`.
        
    Examples
    --------
    Spread secondary structure annotation to every atom of a 20 residue
    peptide (with 304 atoms).
    
        >>> sse = annotate_sse(atom_array, "A")
        >>> print(len(sse))
        20
        >>> print(sse)
        ['c' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c']
        >>> atom_wise_sse = spread_residue_wise(atom_array, sse)
        >>> print(len(atom_wise_sse))
        304
        >>> print(atom_wise_sse)
        ['c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a'
         'a' 'a' 'a' 'a' 'a' 'a' 'a' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c'
         'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c' 'c']
    """
    starts = get_residue_starts(array)
    output_data = np.zeros(array.array_length(), dtype=input_data.dtype)
    for i in range(len(starts)-1):
        output_data[starts[i]:starts[i+1]] = input_data[i]
    output_data[starts[-1]:] = input_data[-1]
    return output_data


def get_residues(array):
    """
    Get the residue IDs and names of an atom array (stack).
    
    The residues are listed in the same order they occur in the array
    (stack).
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack), where the residues are determined.
        
    Returns
    -------
    ids : ndarray, dtype=int
        List of residue IDs.
    names : ndarray, dtype="U3"
        List of residue names.
        
    Examples
    --------
    Get the residue names of a 20 residue peptide.
    
        >>> print(atom_array.res_name)
        ['ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'LEU' 'LEU' 'LEU' 'LEU'
         'LEU' 'LEU' 'LEU' 'LEU' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR'
         'TYR' 'TYR' 'TYR' 'TYR' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE'
         'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'TRP' 'TRP' 'TRP'
         'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'LEU'
         'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS'
         'LYS' 'LYS' 'LYS' 'LYS' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP'
         'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER'
         'SER' 'SER' 'SER' 'GLY' 'GLY' 'GLY' 'GLY' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG'
         'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER']
        >>> ids, names = get_residues(atom_array)
        >>> print(names)
        ['ASN' 'LEU' 'TYR' 'ILE' 'GLN' 'TRP' 'LEU' 'LYS' 'ASP' 'GLY' 'GLY' 'PRO'
         'SER' 'SER' 'GLY' 'ARG' 'PRO' 'PRO' 'PRO' 'SER']
    """
    starts = get_residue_starts(array)
    return array.res_id[starts], array.res_name[starts]

def get_residue_count(array):
    """
    Get the amount of residues in an atom array (stack).
    
    The count is determined from the `res_id` and `chain_id` annotation.
    Each time the residue ID or chain ID changes,
    the count is incremented. Special rules apply to hetero residues
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack), where the residues are counted.
        
    Returns
    -------
    count : int
        Amount of residues.
    """
    return len(get_residue_starts(array))
