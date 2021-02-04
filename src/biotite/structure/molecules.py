# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides utility for separating structures into single
molecules.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["get_molecule_indices", "get_molecule_masks", "molecule_iter"]

import numpy as np
from .atoms import AtomArray, AtomArrayStack
from .bonds import BondList, find_connected


def get_molecule_indices(array):
    """
    Get an index array for each molecule in the given structure.

    A molecule is defined as a group of atoms that are directly or
    indirectly connected via covalent bonds.
    In this function a single atom, that has no connection to any other
    atom (e.g. an ion), also qualifies as a molecule.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack or BondList
        The input structure with an associated :class:`BondList`.
        Alternatively, the :class:`BondList` can be directly supplied.
    
    Returns
    -------
    indices : list of ndarray, dtype=int
        Each element in the list is an index array referring to the
        atoms of a single molecule.
        Consequently, the length of this list is equal to the number of
        molecules in the input `array`.
    
    See also
    --------
    get_molecule_masks
    molecule_iter

    Examples
    --------
    Get an :class:`AtomArray` for ATP and show that it is a single
    molecule:

    >>> atp = residue("ATP")
    >>> indices = get_molecule_indices(atp)
    >>> print(len(indices))
    1

    Separate ATP into two molecules by breaking the glycosidic bond
    to the triphosphate:

    >>> i, j = np.where(np.isin(atp.atom_name, ("O5'", "PA")))[0]
    >>> atp.bonds.remove_bond(i, j)
    >>> indices = get_molecule_indices(atp)
    >>> print(len(indices))
    2
    >>> print(atp[indices[0]])
    HET         0  ATP PG     P         1.200   -0.226   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.672
    HET         0  ATP O2G    O         2.123   -1.036   -7.891
    HET         0  ATP O3G    O        -0.302   -0.139   -7.421
    HET         0  ATP PB     P         0.255   -0.130   -4.446
    HET         0  ATP O1B    O         0.810    1.234   -4.304
    HET         0  ATP O2B    O        -1.231   -0.044   -5.057
    HET         0  ATP O3B    O         1.192   -0.990   -5.433
    HET         0  ATP PA     P        -0.745    0.068   -2.071
    HET         0  ATP O1A    O        -2.097    0.143   -2.669
    HET         0  ATP O2A    O        -0.125    1.549   -1.957
    HET         0  ATP O3A    O         0.203   -0.840   -3.002
    HET         0  ATP HOG2   H         2.100   -0.546   -8.725
    HET         0  ATP HOG3   H        -0.616   -1.048   -7.522
    HET         0  ATP HOB2   H        -1.554   -0.952   -5.132
    HET         0  ATP HOA2   H         0.752    1.455   -1.563
    >>> print(atp[indices[1]])
    HET         0  ATP O5'    O        -0.844   -0.587   -0.604
    HET         0  ATP C5'    C        -1.694    0.260    0.170
    HET         0  ATP C4'    C        -1.831   -0.309    1.584
    HET         0  ATP O4'    O        -0.542   -0.355    2.234
    HET         0  ATP C3'    C        -2.683    0.630    2.465
    HET         0  ATP O3'    O        -4.033    0.165    2.534
    HET         0  ATP C2'    C        -2.011    0.555    3.856
    HET         0  ATP O2'    O        -2.926    0.043    4.827
    HET         0  ATP C1'    C        -0.830   -0.418    3.647
    HET         0  ATP N9     N         0.332    0.015    4.425
    HET         0  ATP C8     C         1.302    0.879    4.012
    HET         0  ATP N7     N         2.184    1.042    4.955
    HET         0  ATP C5     C         1.833    0.300    6.033
    HET         0  ATP C6     C         2.391    0.077    7.303
    HET         0  ATP N6     N         3.564    0.706    7.681
    HET         0  ATP N1     N         1.763   -0.747    8.135
    HET         0  ATP C2     C         0.644   -1.352    7.783
    HET         0  ATP N3     N         0.088   -1.178    6.602
    HET         0  ATP C4     C         0.644   -0.371    5.704
    HET         0  ATP H5'1   H        -2.678    0.312   -0.296
    HET         0  ATP H5'2   H        -1.263    1.259    0.221
    HET         0  ATP H4'    H        -2.275   -1.304    1.550
    HET         0  ATP H3'    H        -2.651    1.649    2.078
    HET         0  ATP HO3'   H        -4.515    0.788    3.094
    HET         0  ATP H2'    H        -1.646    1.537    4.157
    HET         0  ATP HO2'   H        -3.667    0.662    4.867
    HET         0  ATP H1'    H        -1.119   -1.430    3.931
    HET         0  ATP H8     H         1.334    1.357    3.044
    HET         0  ATP HN61   H         3.938    0.548    8.562
    HET         0  ATP HN62   H         4.015    1.303    7.064
    HET         0  ATP H2     H         0.166   -2.014    8.490
    """
    if isinstance(array, BondList):
        bonds = array
    elif isinstance(array, (AtomArray, AtomArrayStack)):
        if array.bonds is None:
            raise ValueError("An associated BondList is required")
        bonds = array.bonds
    else:
        raise TypeError(
            f"Expected a 'BondList', 'AtomArray' or 'AtomArrayStack', "
            f"not '{type(array).__name__}'"
        )
    
    molecule_indices = []
    visited_mask = np.zeros(bonds.get_atom_count(), dtype=bool)
    while not visited_mask.all():
        root = np.argmin(visited_mask)
        connected = find_connected(bonds, root)
        visited_mask[connected] = True
        molecule_indices.append(connected)
    return molecule_indices


def get_molecule_masks(array):
    """
    Get a boolean mask for each molecule in the given structure.

    A molecule is defined as a group of atoms that are directly or
    indirectly connected via covalent bonds.
    In this function a single atom, that has no connection to any other
    atom (e.g. an ion), also qualifies as a molecule.

    Parameters
    ----------
    array : AtomArray, shape=(n,) or AtomArrayStack, shape=(m,n) or BondList
        The input structure with an associated :class:`BondList`.
        Alternatively, the :class:`BondList` can be directly supplied.
    
    Returns
    -------
    masks : ndarray, shape=(k,n), dtype=bool, 
        Each element in the array is a boolean mask referring to the
        atoms of a single molecule.
        Consequently, the length of this list is equal to the number of
        molecules in the input `array`.
    
    See also
    --------
    get_molecule_indices
    molecule_iter

    Examples
    --------
    Get an :class:`AtomArray` for ATP and show that it is a single
    molecule:

    >>> atp = residue("ATP")
    >>> masks = get_molecule_masks(atp)
    >>> print(len(masks))
    1

    Separate ATP into two molecules by breaking the glycosidic bond
    to the triphosphate:

    >>> i, j = np.where(np.isin(atp.atom_name, ("O5'", "PA")))[0]
    >>> atp.bonds.remove_bond(i, j)
    >>> masks = get_molecule_masks(atp)
    >>> print(len(masks))
    2
    >>> print(atp[masks[0]])
    HET         0  ATP PG     P         1.200   -0.226   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.672
    HET         0  ATP O2G    O         2.123   -1.036   -7.891
    HET         0  ATP O3G    O        -0.302   -0.139   -7.421
    HET         0  ATP PB     P         0.255   -0.130   -4.446
    HET         0  ATP O1B    O         0.810    1.234   -4.304
    HET         0  ATP O2B    O        -1.231   -0.044   -5.057
    HET         0  ATP O3B    O         1.192   -0.990   -5.433
    HET         0  ATP PA     P        -0.745    0.068   -2.071
    HET         0  ATP O1A    O        -2.097    0.143   -2.669
    HET         0  ATP O2A    O        -0.125    1.549   -1.957
    HET         0  ATP O3A    O         0.203   -0.840   -3.002
    HET         0  ATP HOG2   H         2.100   -0.546   -8.725
    HET         0  ATP HOG3   H        -0.616   -1.048   -7.522
    HET         0  ATP HOB2   H        -1.554   -0.952   -5.132
    HET         0  ATP HOA2   H         0.752    1.455   -1.563
    >>> print(atp[masks[1]])
    HET         0  ATP O5'    O        -0.844   -0.587   -0.604
    HET         0  ATP C5'    C        -1.694    0.260    0.170
    HET         0  ATP C4'    C        -1.831   -0.309    1.584
    HET         0  ATP O4'    O        -0.542   -0.355    2.234
    HET         0  ATP C3'    C        -2.683    0.630    2.465
    HET         0  ATP O3'    O        -4.033    0.165    2.534
    HET         0  ATP C2'    C        -2.011    0.555    3.856
    HET         0  ATP O2'    O        -2.926    0.043    4.827
    HET         0  ATP C1'    C        -0.830   -0.418    3.647
    HET         0  ATP N9     N         0.332    0.015    4.425
    HET         0  ATP C8     C         1.302    0.879    4.012
    HET         0  ATP N7     N         2.184    1.042    4.955
    HET         0  ATP C5     C         1.833    0.300    6.033
    HET         0  ATP C6     C         2.391    0.077    7.303
    HET         0  ATP N6     N         3.564    0.706    7.681
    HET         0  ATP N1     N         1.763   -0.747    8.135
    HET         0  ATP C2     C         0.644   -1.352    7.783
    HET         0  ATP N3     N         0.088   -1.178    6.602
    HET         0  ATP C4     C         0.644   -0.371    5.704
    HET         0  ATP H5'1   H        -2.678    0.312   -0.296
    HET         0  ATP H5'2   H        -1.263    1.259    0.221
    HET         0  ATP H4'    H        -2.275   -1.304    1.550
    HET         0  ATP H3'    H        -2.651    1.649    2.078
    HET         0  ATP HO3'   H        -4.515    0.788    3.094
    HET         0  ATP H2'    H        -1.646    1.537    4.157
    HET         0  ATP HO2'   H        -3.667    0.662    4.867
    HET         0  ATP H1'    H        -1.119   -1.430    3.931
    HET         0  ATP H8     H         1.334    1.357    3.044
    HET         0  ATP HN61   H         3.938    0.548    8.562
    HET         0  ATP HN62   H         4.015    1.303    7.064
    HET         0  ATP H2     H         0.166   -2.014    8.490
    """
    if isinstance(array, BondList):
        bonds = array
    elif isinstance(array, (AtomArray, AtomArrayStack)):
        if array.bonds is None:
            raise ValueError("An associated BondList is required")
        bonds = array.bonds
    else:
        raise TypeError(
            f"Expected a 'BondList', 'AtomArray' or 'AtomArrayStack', "
            f"not '{type(array).__name__}'"
        )
    
    molecule_indices = get_molecule_indices(bonds)
    molecule_masks = np.zeros(
        (len(molecule_indices), bonds.get_atom_count()),
        dtype=bool
    )
    for i in range(len(molecule_indices)):
        molecule_masks[i, molecule_indices[i]] = True
    return molecule_masks


def molecule_iter(array):
    """
    Iterate over each molecule in a input structure.

    A molecule is defined as a group of atoms that are directly or
    indirectly connected via covalent bonds.
    In this function a single atom, that has no connection to any other
    atom (e.g. an ion), also qualifies as a molecule.

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The input structure with an associated :class:`BondList`.
    
    Yields
    ------
    molecule : AtomArray or AtomArrayStack
        A single molecule of the input `array`.
    
    See also
    --------
    get_molecule_indices
    get_molecule_masks

    Examples
    --------
    Get an :class:`AtomArray` for ATP and break it into two molecules
    at the glycosidic bond to the triphosphate:

    >>> atp = residue("ATP")
    >>> i, j = np.where(np.isin(atp.atom_name, ("O5'", "PA")))[0]
    >>> atp.bonds.remove_bond(i, j)
    >>> for molecule in molecule_iter(atp):
    ...     print("New molecule")
    ...     print(molecule)
    ...     print()
    New molecule
    HET         0  ATP PG     P         1.200   -0.226   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.672
    HET         0  ATP O2G    O         2.123   -1.036   -7.891
    HET         0  ATP O3G    O        -0.302   -0.139   -7.421
    HET         0  ATP PB     P         0.255   -0.130   -4.446
    HET         0  ATP O1B    O         0.810    1.234   -4.304
    HET         0  ATP O2B    O        -1.231   -0.044   -5.057
    HET         0  ATP O3B    O         1.192   -0.990   -5.433
    HET         0  ATP PA     P        -0.745    0.068   -2.071
    HET         0  ATP O1A    O        -2.097    0.143   -2.669
    HET         0  ATP O2A    O        -0.125    1.549   -1.957
    HET         0  ATP O3A    O         0.203   -0.840   -3.002
    HET         0  ATP HOG2   H         2.100   -0.546   -8.725
    HET         0  ATP HOG3   H        -0.616   -1.048   -7.522
    HET         0  ATP HOB2   H        -1.554   -0.952   -5.132
    HET         0  ATP HOA2   H         0.752    1.455   -1.563
    <BLANKLINE>
    New molecule
    HET         0  ATP O5'    O        -0.844   -0.587   -0.604
    HET         0  ATP C5'    C        -1.694    0.260    0.170
    HET         0  ATP C4'    C        -1.831   -0.309    1.584
    HET         0  ATP O4'    O        -0.542   -0.355    2.234
    HET         0  ATP C3'    C        -2.683    0.630    2.465
    HET         0  ATP O3'    O        -4.033    0.165    2.534
    HET         0  ATP C2'    C        -2.011    0.555    3.856
    HET         0  ATP O2'    O        -2.926    0.043    4.827
    HET         0  ATP C1'    C        -0.830   -0.418    3.647
    HET         0  ATP N9     N         0.332    0.015    4.425
    HET         0  ATP C8     C         1.302    0.879    4.012
    HET         0  ATP N7     N         2.184    1.042    4.955
    HET         0  ATP C5     C         1.833    0.300    6.033
    HET         0  ATP C6     C         2.391    0.077    7.303
    HET         0  ATP N6     N         3.564    0.706    7.681
    HET         0  ATP N1     N         1.763   -0.747    8.135
    HET         0  ATP C2     C         0.644   -1.352    7.783
    HET         0  ATP N3     N         0.088   -1.178    6.602
    HET         0  ATP C4     C         0.644   -0.371    5.704
    HET         0  ATP H5'1   H        -2.678    0.312   -0.296
    HET         0  ATP H5'2   H        -1.263    1.259    0.221
    HET         0  ATP H4'    H        -2.275   -1.304    1.550
    HET         0  ATP H3'    H        -2.651    1.649    2.078
    HET         0  ATP HO3'   H        -4.515    0.788    3.094
    HET         0  ATP H2'    H        -1.646    1.537    4.157
    HET         0  ATP HO2'   H        -3.667    0.662    4.867
    HET         0  ATP H1'    H        -1.119   -1.430    3.931
    HET         0  ATP H8     H         1.334    1.357    3.044
    HET         0  ATP HN61   H         3.938    0.548    8.562
    HET         0  ATP HN62   H         4.015    1.303    7.064
    HET         0  ATP H2     H         0.166   -2.014    8.490
    <BLANKLINE>
    """
    if array.bonds is None:
        raise ValueError("An associated BondList is required")
    bonds = array.bonds
    
    visited_mask = np.zeros(bonds.get_atom_count(), dtype=bool)
    while not visited_mask.all():
        # Take the first atom that has not been considered yet as root
        root = np.argmin(visited_mask)
        connected = find_connected(bonds, root)
        visited_mask[connected] = True
        yield array[..., connected]
