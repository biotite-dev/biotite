# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

"""
This module provides utility for handling data on residue level, rather than
atom level.
"""

import numpy as np
from .atoms import AtomArray, AtomArrayStack

__all__ = ["apply_residue_wise", "get_residues"]


def apply_residue_wise(array, data, function, axis=None):
    """
    Apply a function to intervals of data, where each interval
    correspond to one residue.
    
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
        have same length as `array.res_id`.
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
        154
        >>> sasa_per_residue = struc.apply_residue_wise(atom_array,
                                                        sasa_per_atom,
                                                        np.sum)
        >>> print(len(sasa_per_residue))
        20
        >>> print(sasa_per_residue)
        [ 157.86574713  118.14982076   94.14992528  114.91493537  113.02598368
           22.97896584   88.05321235  145.11664892   59.42074947   37.8765113
            0.          114.62239027  106.00488852   28.77275384   91.53945028
          115.09834155  113.25537021   69.12604652   48.9054493   179.75047111]

    Calculate the controids of each residue for the same peptide.
    
        >>> print(len(atom_array))
        154
        >>> centroids = apply_residue_wise(atom_array, atom_array.coord,
        ...                                      np.average, axis=0)
        >>> print(len(centroids))
        20
        >>> print(centroids)
        [[ -9.33587497e+00   3.08837497e+00  -2.04937501e+00]
         [ -4.62474999e+00   4.99312496e+00  -1.83012499e+00]
         [ -2.44825002e+00   2.78391666e+00   3.04791670e+00]
         [ -6.55874997e+00  -5.59624996e-01   6.70249989e-01]
         [ -4.45077781e+00  -1.42922222e+00  -4.00411113e+00]
         [  1.18471429e+00   2.28571253e-03   1.03314285e+00]
         [ -1.97637503e+00  -2.73912497e+00   2.88100000e+00]
         [ -3.24277778e+00  -5.74677769e+00  -2.27588885e+00]
         [  8.07999995e-01  -5.58362508e+00  -2.71987501e+00]
         [  2.34224999e+00  -5.97549999e+00   1.90449998e+00]
         [  3.41624999e+00  -3.19624996e+00   2.24575004e+00]
         [  6.13757140e+00  -2.77557147e+00   4.25599994e+00]
         [  6.77583329e+00  -6.59999990e+00   2.68549995e+00]
         [  6.13083331e+00  -4.97816674e+00  -1.50333334e+00]
         [  9.09824991e+00  -2.48450002e+00  -2.16374999e+00]
         [  5.71227275e+00  -1.31709090e+00  -3.11790907e+00]
         [  6.69085721e+00   3.38457145e+00  -5.37428569e-01]
         [  3.91357149e+00   4.58928575e+00   2.10914286e+00]
         [  5.29857130e-01   5.99642863e+00   1.11714291e-01]
         [  7.25857139e-01   1.04195715e+01   9.49285729e-01]]
    
    """
    ids = array.res_id
    # The result array
    processed_data = None
    # Start of each interval
    start = 0
    # Index of last value in result array
    i = 0
    while start in range(len(ids)):
        # End of each interval
        stop = start
        # Determine length of interval
        # When first condition fails the second one is not checked,
        # otherwise the second condition could throw IndexError
        while stop < len(ids) and ids[stop] == ids[start]:
            stop += 1
        interval = data[start:stop]
        if axis == None:
            value = function(interval)
        else:
            value = function(interval, axis=axis)
        # Identify the shape of the resulting array by evaluation
        # of the function return value for the first interval
        if processed_data is None:
            if isinstance(value, np.ndarray):
                # Maximum length of the processed data
                # is length of interval of size 1 -> length of all IDs
                processed_data = np.zeros((len(ids),) + value.shape,
                                          dtype=value.dtype)
            else:
                # Scalar value -> one dimensional result array
                # Maximum length of the processed data
                # is length of interval of size 1 -> length of all IDs
                processed_data = np.zeros(len(ids),
                                          dtype=type(value))
        # Write values into result arrays
        processed_data[i] = value
        start = stop
        i += 1
    # Trim result array to correct size
    return processed_data[:i]


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
    ids : ndarray(dtype=int)
        List of residue IDs.
    names : ndarray(dtype="U3")
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
        >>> ids, names = struc.get_residues(atom_array)
        >>> print(names)
        ['ASN' 'LEU' 'TYR' 'ILE' 'GLN' 'TRP' 'LEU' 'LYS' 'ASP' 'GLY' 'GLY' 'PRO'
         'SER' 'SER' 'GLY' 'ARG' 'PRO' 'PRO' 'PRO' 'SER']
    """
    # Maximum length is length of atom array
    ids = np.zeros(len(array.res_id), dtype=int)
    names = np.zeros(len(array.res_id), dtype="U3")
    ids[0] = array.res_id[0]
    names[0] = array.res_name[0]
    i = 1
    for j in range(len(array.res_id)):
        if array.res_id[j] != ids[i-1]:
            ids[i] = array.res_id[j]
            names[i] = array.res_name[j]
            i += 1
    # Trim to correct size
    return ids[:i], names[:i]
            