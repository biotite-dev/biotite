# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility for handling data on residue level, rather than
atom level.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = [
    "get_residue_starts",
    "apply_residue_wise",
    "spread_residue_wise",
    "get_residue_masks",
    "get_residue_starts_for",
    "get_residue_positions",
    "get_all_residue_positions",
    "get_residues",
    "get_residue_count",
    "residue_iter",
    "get_atom_name_indices",
]

import numpy as np
from biotite.structure.segments import (
    apply_segment_wise,
    get_all_segment_positions,
    get_segment_masks,
    get_segment_positions,
    get_segment_starts,
    get_segment_starts_for,
    segment_iter,
    spread_segment_wise,
)


def get_residue_starts(array, add_exclusive_stop=False, extra_categories=()):
    """
    Get indices for an atom array, each indicating the beginning of
    a residue.

    A new residue starts, either when the chain ID, sym ID, residue ID,
    insertion code or residue name changes from one to the next atom.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to get the residue starts from.
    add_exclusive_stop : bool, optional
        If true, the exclusive stop of the input atom array, i.e.
        ``array.array_length()``, is added to the returned array of
        start indices as last element.
    extra_categories : tuple of str, optional
        Additional annotation categories that induce the start of a new residue,
        when their value change from one atom to the next.

    Returns
    -------
    starts : ndarray, dtype=int
        The start indices of residues in `array`.

    Notes
    -----
    This method is internally used by all other residue-related
    functions.

    Examples
    --------

    >>> print(get_residue_starts(atom_array))
    [  0  16  35  56  75  92 116 135 157 169 176 183 197 208 219 226 250 264
     278 292]
    >>> print(get_residue_starts(atom_array, add_exclusive_stop=True))
    [  0  16  35  56  75  92 116 135 157 169 176 183 197 208 219 226 250 264
     278 292 304]
    """
    categories = ["chain_id", "res_id", "ins_code"] + list(extra_categories)
    if "sym_id" in array.get_annotation_categories():
        categories.append("sym_id")
    return get_segment_starts(array, add_exclusive_stop, equal_categories=categories)


def apply_residue_wise(array, data, function, axis=None):
    """
    Apply a function to intervals of data, where each interval
    corresponds to one residue.

    The function takes an atom array (stack) and an data array
    (`ndarray`) of the same length. The function iterates through the
    residue IDs of the atom array (stack) and identifies intervals of
    the same ID. Then the data is
    partitioned into the same intervals, and each interval (also an
    :class:`ndarray`) is put as parameter into `function`. Each return value is
    stored as element in the resulting :class:`ndarray`, therefore each element
    corresponds to one residue.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.
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
        Residue-wise evaluation of `data` by `function`. The size of the first dimension
        of this array is equal to the amount of residues.

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
    [157.979 117.136  94.983 115.485 113.583  23.471  93.013 144.173  61.561
      38.885   0.792 114.053 108.568  27.888  83.583 113.016 114.318  74.281
      47.811 172.035]

    Calculate the centroids of each residue for the same peptide.

    >>> print(len(atom_array))
    304
    >>> centroids = apply_residue_wise(atom_array, atom_array.coord,
    ...                                np.average, axis=0)
    >>> print(len(centroids))
    20
    >>> print(centroids)
    [[-9.582  3.378 -2.073]
     [-4.670  5.816 -1.860]
     [-2.461  3.060  3.076]
     [-7.211 -0.396  1.013]
     [-4.698 -1.080 -4.284]
     [ 1.172  0.206  1.038]
     [-2.160 -2.245  3.541]
     [-3.682 -5.540 -2.895]
     [ 0.711 -5.409 -2.549]
     [ 2.002 -6.322  1.695]
     [ 2.799 -3.140  2.327]
     [ 5.901 -2.489  4.845]
     [ 6.754 -6.712  3.094]
     [ 5.699 -5.101 -1.209]
     [ 9.295 -2.970 -1.835]
     [ 5.518 -1.521 -3.473]
     [ 7.219  3.673 -0.684]
     [ 4.007  4.364  2.674]
     [ 0.341  5.575 -0.254]
     [ 1.194 10.416  1.130]]
    """
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return apply_segment_wise(starts, data, function, axis)


def spread_residue_wise(array, input_data):
    """
    Expand residue-wise data to atom-wise data.

    Each value in the residue-wise input is assigned to all atoms of
    this residue:

    ``output_data[i] = input_data[j]``,
    *i* is incremented from atom to atom,
    *j* is incremented every residue change.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.
    input_data : ndarray
        The data to be spread.
        The length of the 0-th axis must be equal to the amount of different residue IDs
        in `array`.

    Returns
    -------
    output_data : ndarray
        Residue-wise spread `input_data`.
        Length is the same as `array_length()` of `array`.

    Examples
    --------
    Spread secondary structure annotation to every atom of a 20 residue
    peptide (with 304 atoms).

        >>> sse = annotate_sse(atom_array)
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
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return spread_segment_wise(starts, input_data)


def get_residue_masks(array, indices):
    """
    Get boolean masks indicating the residues to which the given atom
    indices belong.

    Parameters
    ----------
    array : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n)
        The atom array (stack) to determine the residues from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices indicate the atoms to get the corresponding
        residues for.
        Negative indices are not allowed.

    Returns
    -------
    residues_masks : ndarray, dtype=bool, shape=(k,n)
        Multiple boolean masks, one for each given index in `indices`.
        Each array masks the atoms that belong to the same residue as
        the atom at the given index.

    Examples
    --------

    >>> indices = [5, 42]
    >>> residue_masks = get_residue_masks(atom_array, indices)
    >>> print(atom_array[indices[0]])
        A       1  ASN CG     C       -10.915    3.130   -2.611
    >>> print(atom_array[residue_masks[0]])
        A       1  ASN N      N        -8.901    4.127   -0.555
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN H1     H        -8.330    3.957    0.261
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HD22   H       -12.757    3.183   -3.294
    >>> print(atom_array[indices[1]])
        A       3  TYR CD2    C        -1.820    4.326    3.332
    >>> print(atom_array[residue_masks[1]])
        A       3  TYR N      N        -4.354    3.455   -0.111
        A       3  TYR CA     C        -3.690    2.738    0.981
        A       3  TYR C      C        -4.102    1.256    1.074
        A       3  TYR O      O        -3.291    0.409    1.442
        A       3  TYR CB     C        -3.964    3.472    2.302
        A       3  TYR CG     C        -2.824    3.339    3.290
        A       3  TYR CD1    C        -2.746    2.217    4.138
        A       3  TYR CD2    C        -1.820    4.326    3.332
        A       3  TYR CE1    C        -1.657    2.076    5.018
        A       3  TYR CE2    C        -0.725    4.185    4.205
        A       3  TYR CZ     C        -0.639    3.053    5.043
        A       3  TYR OH     O         0.433    2.881    5.861
        A       3  TYR H      H        -4.934    4.245    0.120
        A       3  TYR HA     H        -2.615    2.768    0.796
        A       3  TYR HB2    H        -4.117    4.513    2.091
        A       3  TYR HB3    H        -4.886    3.096    2.750
        A       3  TYR HD1    H        -3.513    1.456    4.101
        A       3  TYR HD2    H        -1.877    5.200    2.695
        A       3  TYR HE1    H        -1.576    1.221    5.669
        A       3  TYR HE2    H         0.033    4.952    4.233
        A       3  TYR HH     H         1.187    3.395    5.567
    """
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_segment_masks(starts, indices)


def get_residue_starts_for(array, indices):
    """
    For each given atom index, get the index that points to the
    start of the residue that atom belongs to.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices point to the atoms to get the corresponding
        residue starts for.
        Negative indices are not allowed.

    Returns
    -------
    start_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the residue starts for the input
        `indices`.

    Examples
    --------

    >>> indices = [5, 42]
    >>> residue_starts = get_residue_starts_for(atom_array, indices)
    >>> print(residue_starts)
    [ 0 35]
    >>> print(atom_array[indices[0]])
        A       1  ASN CG     C       -10.915    3.130   -2.611
    >>> print(atom_array[residue_starts[0]])
        A       1  ASN N      N        -8.901    4.127   -0.555
    >>> print(atom_array[indices[1]])
        A       3  TYR CD2    C        -1.820    4.326    3.332
    >>> print(atom_array[residue_starts[1]])
        A       3  TYR N      N        -4.354    3.455   -0.111
    """
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_segment_starts_for(starts, indices)


def get_residue_positions(array, indices):
    """
    For each given atom index, obtain the position of the residue
    corresponding to this index in the input `array`.

    For example, the position of the first residue in the atom array is
    ``0``, the the position of the second residue is ``1``, etc.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.
    indices : ndarray, dtype=int, shape=(k,)
        These indices point to the atoms to get the corresponding
        residue positions for.
        Negative indices are not allowed.

    Returns
    -------
    residue_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the position of the residues.

    See Also
    --------
    get_all_residue_positions :
        Similar to this function, but for all atoms in the :class:`struc.AtomArray`.

    Examples
    --------
    >>> atom_index = [5, 42]
    >>> print(atom_array.res_name[atom_index])
    ['ASN' 'TYR']
    >>> _, residues = get_residues(atom_array)
    >>> print(residues)
    ['ASN' 'LEU' 'TYR' 'ILE' 'GLN' 'TRP' 'LEU' 'LYS' 'ASP' 'GLY' 'GLY' 'PRO'
     'SER' 'SER' 'GLY' 'ARG' 'PRO' 'PRO' 'PRO' 'SER']
    >>> residue_index = get_residue_positions(atom_array, atom_index)
    >>> print(residue_index)
    [0 2]
    >>> print(residues[residue_index])
    ['ASN' 'TYR']
    """
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_segment_positions(starts, indices)


def get_all_residue_positions(array):
    """
    For each atom, obtain the position of the residue
    corresponding to this atom in the input `array`.

    For example, the position of the first residue in the atom array is
    ``0``, the the position of the second residue is ``1``, etc.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.

    Returns
    -------
    residue_indices : ndarray, dtype=int, shape=(k,)
        The indices that point to the position of the residues.

    See Also
    --------
    get_residue_positions :
        Similar to this function, but for a given subset of atom indices.

    Examples
    --------
    >>> print(get_all_residue_positions(atom_array))
    [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1
      1  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2  2  2  2  2  2
      2  2  2  2  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
      3  3  3  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  5  5  5  5
      5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  6  6  6  6
      6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  7  7  7  7  7  7  7  7  7
      7  7  7  7  7  7  7  7  7  7  7  7  7  8  8  8  8  8  8  8  8  8  8  8
      8  9  9  9  9  9  9  9 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11
     11 11 11 11 11 12 12 12 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13
     13 13 13 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15 15 15 15 15 15
     15 15 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16
     17 17 17 17 17 17 17 17 17 17 17 17 17 17 18 18 18 18 18 18 18 18 18 18
     18 18 18 18 19 19 19 19 19 19 19 19 19 19 19 19]
    """
    starts = get_residue_starts(array, add_exclusive_stop=True)
    return get_all_segment_positions(starts, array.array_length())


def get_residues(array):
    """
    Get the residue IDs and names of an atom array (stack).

    The residues are listed in the same order they occur in the array
    (stack).

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.

    Returns
    -------
    ids : ndarray, dtype=int
        List of residue IDs.
    names : ndarray, dtype="U5"
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
    the count is incremented. Special rules apply to hetero residues.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to determine the residues from.

    Returns
    -------
    count : int
        Amount of residues.
    """
    return len(get_residue_starts(array))


def residue_iter(array):
    """
    Iterate over all residues in an atom array (stack).

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The atom array (stack) to iterate over.

    Yields
    ------
    residue : AtomArray or AtomArrayStack
        A single residue of the input `array`.

    Examples
    --------

    >>> for res in residue_iter(atom_array[:50]):
    ...     print("New residue")
    ...     print(res)
    ...     print()
    New residue
        A       1  ASN N      N        -8.901    4.127   -0.555
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN H1     H        -8.330    3.957    0.261
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HD22   H       -12.757    3.183   -3.294
    <BLANKLINE>
    New residue
        A       2  LEU N      N        -6.379    4.031   -2.228
        A       2  LEU CA     C        -4.923    4.002   -2.452
        A       2  LEU C      C        -4.136    3.187   -1.404
        A       2  LEU O      O        -3.391    2.274   -1.760
        A       2  LEU CB     C        -4.411    5.450   -2.619
        A       2  LEU CG     C        -4.795    6.450   -1.495
        A       2  LEU CD1    C        -3.612    6.803   -0.599
        A       2  LEU CD2    C        -5.351    7.748   -2.084
        A       2  LEU H      H        -6.821    4.923   -2.394
        A       2  LEU HA     H        -4.750    3.494   -3.403
        A       2  LEU HB2    H        -3.340    5.414   -2.672
        A       2  LEU HB3    H        -4.813    5.817   -3.564
        A       2  LEU HG     H        -5.568    6.022   -0.858
        A       2  LEU HD11   H        -3.207    5.905   -0.146
        A       2  LEU HD12   H        -2.841    7.304   -1.183
        A       2  LEU HD13   H        -3.929    7.477    0.197
        A       2  LEU HD21   H        -4.607    8.209   -2.736
        A       2  LEU HD22   H        -6.255    7.544   -2.657
        A       2  LEU HD23   H        -5.592    8.445   -1.281
    <BLANKLINE>
    New residue
        A       3  TYR N      N        -4.354    3.455   -0.111
        A       3  TYR CA     C        -3.690    2.738    0.981
        A       3  TYR C      C        -4.102    1.256    1.074
        A       3  TYR O      O        -3.291    0.409    1.442
        A       3  TYR CB     C        -3.964    3.472    2.302
        A       3  TYR CG     C        -2.824    3.339    3.290
        A       3  TYR CD1    C        -2.746    2.217    4.138
        A       3  TYR CD2    C        -1.820    4.326    3.332
        A       3  TYR CE1    C        -1.657    2.076    5.018
        A       3  TYR CE2    C        -0.725    4.185    4.205
        A       3  TYR CZ     C        -0.639    3.053    5.043
        A       3  TYR OH     O         0.433    2.881    5.861
        A       3  TYR H      H        -4.934    4.245    0.120
        A       3  TYR HA     H        -2.615    2.768    0.796
        A       3  TYR HB2    H        -4.117    4.513    2.091
    <BLANKLINE>
    """
    # The exclusive stop is appended to the residue starts
    starts = get_residue_starts(array, add_exclusive_stop=True)
    for residue in segment_iter(array, starts):
        yield residue


def get_atom_name_indices(atoms, atom_names):
    """
    For each residue, get the index of the atom with the given atom name.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        Search for the indices of the given atom names in this structure.
    atom_names : list of str, length=p
        The names of the atoms to get the indices of.

    Returns
    -------
    indices : ndarray, dtype=int, shape=(k, p)
        For every residue and atom name, the return value contains the atom index in
        the :class:`AtomArray` where the sought atom name is located.
        Where the atom name is not present in a residue, the array is filled with `-1`.

    Examples
    --------

    >>> indices = get_atom_name_indices(atom_array, ["CA", "CB"])
    >>> print(indices)
    [[  1   4]
     [ 17  20]
     [ 36  39]
     [ 57  60]
     [ 76  79]
     [ 93  96]
     [117 120]
     [136 139]
     [158 161]
     [170  -1]
     [177  -1]
     [184 187]
     [198 201]
     [209 212]
     [220  -1]
     [227 230]
     [251 254]
     [265 268]
     [279 282]
     [293 296]]
    >>> for row in indices:
    ...     for index in row:
    ...         if index != -1:
    ...             print(atom_array[index])
    ...     print()
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN CB     C        -9.437    3.396   -2.889
    <BLANKLINE>
        A       2  LEU CA     C        -4.923    4.002   -2.452
        A       2  LEU CB     C        -4.411    5.450   -2.619
    <BLANKLINE>
        A       3  TYR CA     C        -3.690    2.738    0.981
        A       3  TYR CB     C        -3.964    3.472    2.302
    <BLANKLINE>
        A       4  ILE CA     C        -5.857   -0.449    0.613
        A       4  ILE CB     C        -7.386   -0.466    0.343
    <BLANKLINE>
        A       5  GLN CA     C        -4.122   -1.167   -2.743
        A       5  GLN CB     C        -4.292   -0.313   -4.013
    <BLANKLINE>
        A       6  TRP CA     C        -0.716   -0.631   -0.993
        A       6  TRP CB     C        -0.221    0.703   -0.417
    <BLANKLINE>
        A       7  LEU CA     C        -1.641   -2.932    1.963
        A       7  LEU CB     C        -2.710   -2.645    3.033
    <BLANKLINE>
        A       8  LYS CA     C        -3.024   -5.791   -0.269
        A       8  LYS CB     C        -4.224   -5.697   -1.232
    <BLANKLINE>
        A       9  ASP CA     C         0.466   -6.016   -1.905
        A       9  ASP CB     C         1.033   -4.839   -2.724
    <BLANKLINE>
        A      10  GLY CA     C         2.060   -6.618    1.593
    <BLANKLINE>
        A      11  GLY CA     C         2.626   -2.967    2.723
    <BLANKLINE>
        A      12  PRO CA     C         6.333   -2.533    3.806
        A      12  PRO CB     C         6.740   -2.387    5.279
    <BLANKLINE>
        A      13  SER CA     C         7.049   -6.179    2.704
        A      13  SER CB     C         6.458   -7.371    3.472
    <BLANKLINE>
        A      14  SER CA     C         6.389   -5.315   -1.015
        A      14  SER CB     C         4.914   -4.993   -1.265
    <BLANKLINE>
        A      15  GLY CA     C         9.451   -3.116   -1.870
    <BLANKLINE>
        A      16  ARG CA     C         7.289    0.084   -2.054
        A      16  ARG CB     C         6.110   -0.243   -2.994
    <BLANKLINE>
        A      17  PRO CA     C         6.782    3.088    0.345
        A      17  PRO CB     C         7.554    4.394    0.119
    <BLANKLINE>
        A      18  PRO CA     C         3.287    4.031    1.686
        A      18  PRO CB     C         3.035    4.190    3.187
    <BLANKLINE>
        A      19  PRO CA     C         1.185    6.543   -0.353
        A      19  PRO CB     C         0.048    6.014   -1.229
    <BLANKLINE>
        A      20  SER CA     C         0.852   10.027    1.285
        A      20  SER CB     C         1.972   11.071    1.284
    <BLANKLINE>
    """
    residue_indices = get_all_residue_positions(atoms)
    indices = np.full(
        (residue_indices[-1] + 1, len(atom_names)), fill_value=-1, dtype=int
    )
    for i, atom_name in enumerate(atom_names):
        if atom_name is None:
            atom_name_indices = np.where(atoms.hetero)[0]
        else:
            atom_name_indices = np.where(atoms.atom_name == atom_name)[0]
        indices[residue_indices[atom_name_indices], i] = atom_name_indices
    return indices
