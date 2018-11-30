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
    [171.01521    111.95127    101.39954    109.50637    116.96978
      21.875536    93.80745    146.43741     69.22964     36.158974
       0.25735924 112.72334    119.67204     28.180838    85.82931
     109.12033    108.86296     70.77379     44.394466   190.57451   ]

    Calculate the controids of each residue for the same peptide.
        
    >>> print(len(atom_array))
    304
    >>> centroids = apply_residue_wise(atom_array, atom_array.coord,
    ...                                np.average, axis=0)
    >>> print(len(centroids))
    20
    >>> print(centroids)
    [[-9.581938    3.3778126  -2.0728126 ]
     [-4.6695266   5.815737   -1.8598946 ]
     [-2.4608097   3.0596666   3.0760477 ]
     [-7.2108426  -0.39636838  1.0131578 ]
     [-4.6978235  -1.0804706  -4.284117  ]
     [ 1.1721249   0.20641668  1.038375  ]
     [-2.1600525  -2.2449472   3.5405266 ]
     [-3.6823182  -5.5397725  -2.8952727 ]
     [ 0.71108335 -5.4094167  -2.5495    ]
     [ 2.0024288  -6.321715    1.6952857 ]
     [ 2.7985713  -3.1399999   2.3274286 ]
     [ 5.9007144  -2.4889286   4.844571  ]
     [ 6.7537274  -6.7123637   3.0941818 ]
     [ 5.6992726  -5.100636   -1.2091817 ]
     [ 9.295427   -2.9695716  -1.8352858 ]
     [ 5.517959   -1.5212501  -3.472667  ]
     [ 7.218929    3.6732144  -0.6843571 ]
     [ 4.006643    4.3640003   2.673857  ]
     [ 0.34114286  5.575286   -0.25428572]
     [ 1.194      10.416249    1.1301666 ]]
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
        ['ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN' 'ASN'
         'ASN' 'ASN' 'ASN' 'ASN' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU'
         'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'TYR'
         'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR'
         'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'TYR' 'ILE' 'ILE' 'ILE' 'ILE'
         'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE' 'ILE'
         'ILE' 'ILE' 'ILE' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN'
         'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'GLN' 'TRP' 'TRP' 'TRP' 'TRP'
         'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP'
         'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'TRP' 'LEU' 'LEU' 'LEU' 'LEU'
         'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU' 'LEU'
         'LEU' 'LEU' 'LEU' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS'
         'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS' 'LYS'
         'LYS' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP' 'ASP'
         'ASP' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY'
         'GLY' 'GLY' 'GLY' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER'
         'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER'
         'SER' 'SER' 'SER' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'GLY' 'ARG' 'ARG'
         'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG'
         'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'ARG' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO' 'PRO'
         'PRO' 'PRO' 'PRO' 'PRO' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER' 'SER'
         'SER' 'SER' 'SER' 'SER']
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
