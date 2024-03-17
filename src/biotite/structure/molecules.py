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
    HET         0  ATP PG     P         1.200   -0.230   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.670
    HET         0  ATP O2G    O         2.120   -1.040   -7.890
    HET         0  ATP O3G    O        -0.300   -0.140   -7.420
    HET         0  ATP PB     P         0.260   -0.130   -4.450
    HET         0  ATP O1B    O         0.810    1.230   -4.300
    HET         0  ATP O2B    O        -1.230   -0.040   -5.060
    HET         0  ATP O3B    O         1.190   -0.990   -5.430
    HET         0  ATP PA     P        -0.740    0.070   -2.070
    HET         0  ATP O1A    O        -2.100    0.140   -2.670
    HET         0  ATP O2A    O        -0.120    1.550   -1.960
    HET         0  ATP O3A    O         0.200   -0.840   -3.000
    HET         0  ATP HOG2   H         2.100   -0.550   -8.730
    HET         0  ATP HOG3   H        -0.620   -1.050   -7.520
    HET         0  ATP HOB2   H        -1.550   -0.950   -5.130
    HET         0  ATP HOA2   H         0.750    1.460   -1.560
    >>> print(atp[indices[1]])
    HET         0  ATP O5'    O        -0.840   -0.590   -0.600
    HET         0  ATP C5'    C        -1.690    0.260    0.170
    HET         0  ATP C4'    C        -1.830   -0.310    1.580
    HET         0  ATP O4'    O        -0.540   -0.360    2.230
    HET         0  ATP C3'    C        -2.680    0.630    2.460
    HET         0  ATP O3'    O        -4.030    0.160    2.530
    HET         0  ATP C2'    C        -2.010    0.560    3.860
    HET         0  ATP O2'    O        -2.930    0.040    4.830
    HET         0  ATP C1'    C        -0.830   -0.420    3.650
    HET         0  ATP N9     N         0.330    0.020    4.430
    HET         0  ATP C8     C         1.300    0.880    4.010
    HET         0  ATP N7     N         2.180    1.040    4.960
    HET         0  ATP C5     C         1.830    0.300    6.030
    HET         0  ATP C6     C         2.390    0.080    7.300
    HET         0  ATP N6     N         3.560    0.710    7.680
    HET         0  ATP N1     N         1.760   -0.750    8.140
    HET         0  ATP C2     C         0.640   -1.350    7.780
    HET         0  ATP N3     N         0.090   -1.180    6.600
    HET         0  ATP C4     C         0.640   -0.370    5.700
    HET         0  ATP H5'1   H        -2.680    0.310   -0.300
    HET         0  ATP H5'2   H        -1.260    1.260    0.220
    HET         0  ATP H4'    H        -2.280   -1.300    1.550
    HET         0  ATP H3'    H        -2.650    1.650    2.080
    HET         0  ATP HO3'   H        -4.520    0.790    3.090
    HET         0  ATP H2'    H        -1.650    1.540    4.160
    HET         0  ATP HO2'   H        -3.670    0.660    4.870
    HET         0  ATP H1'    H        -1.120   -1.430    3.930
    HET         0  ATP H8     H         1.330    1.360    3.040
    HET         0  ATP HN61   H         3.940    0.550    8.560
    HET         0  ATP HN62   H         4.020    1.300    7.060
    HET         0  ATP H2     H         0.170   -2.010    8.490
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
    HET         0  ATP PG     P         1.200   -0.230   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.670
    HET         0  ATP O2G    O         2.120   -1.040   -7.890
    HET         0  ATP O3G    O        -0.300   -0.140   -7.420
    HET         0  ATP PB     P         0.260   -0.130   -4.450
    HET         0  ATP O1B    O         0.810    1.230   -4.300
    HET         0  ATP O2B    O        -1.230   -0.040   -5.060
    HET         0  ATP O3B    O         1.190   -0.990   -5.430
    HET         0  ATP PA     P        -0.740    0.070   -2.070
    HET         0  ATP O1A    O        -2.100    0.140   -2.670
    HET         0  ATP O2A    O        -0.120    1.550   -1.960
    HET         0  ATP O3A    O         0.200   -0.840   -3.000
    HET         0  ATP HOG2   H         2.100   -0.550   -8.730
    HET         0  ATP HOG3   H        -0.620   -1.050   -7.520
    HET         0  ATP HOB2   H        -1.550   -0.950   -5.130
    HET         0  ATP HOA2   H         0.750    1.460   -1.560
    >>> print(atp[masks[1]])
    HET         0  ATP O5'    O        -0.840   -0.590   -0.600
    HET         0  ATP C5'    C        -1.690    0.260    0.170
    HET         0  ATP C4'    C        -1.830   -0.310    1.580
    HET         0  ATP O4'    O        -0.540   -0.360    2.230
    HET         0  ATP C3'    C        -2.680    0.630    2.460
    HET         0  ATP O3'    O        -4.030    0.160    2.530
    HET         0  ATP C2'    C        -2.010    0.560    3.860
    HET         0  ATP O2'    O        -2.930    0.040    4.830
    HET         0  ATP C1'    C        -0.830   -0.420    3.650
    HET         0  ATP N9     N         0.330    0.020    4.430
    HET         0  ATP C8     C         1.300    0.880    4.010
    HET         0  ATP N7     N         2.180    1.040    4.960
    HET         0  ATP C5     C         1.830    0.300    6.030
    HET         0  ATP C6     C         2.390    0.080    7.300
    HET         0  ATP N6     N         3.560    0.710    7.680
    HET         0  ATP N1     N         1.760   -0.750    8.140
    HET         0  ATP C2     C         0.640   -1.350    7.780
    HET         0  ATP N3     N         0.090   -1.180    6.600
    HET         0  ATP C4     C         0.640   -0.370    5.700
    HET         0  ATP H5'1   H        -2.680    0.310   -0.300
    HET         0  ATP H5'2   H        -1.260    1.260    0.220
    HET         0  ATP H4'    H        -2.280   -1.300    1.550
    HET         0  ATP H3'    H        -2.650    1.650    2.080
    HET         0  ATP HO3'   H        -4.520    0.790    3.090
    HET         0  ATP H2'    H        -1.650    1.540    4.160
    HET         0  ATP HO2'   H        -3.670    0.660    4.870
    HET         0  ATP H1'    H        -1.120   -1.430    3.930
    HET         0  ATP H8     H         1.330    1.360    3.040
    HET         0  ATP HN61   H         3.940    0.550    8.560
    HET         0  ATP HN62   H         4.020    1.300    7.060
    HET         0  ATP H2     H         0.170   -2.010    8.490
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
    HET         0  ATP PG     P         1.200   -0.230   -6.850
    HET         0  ATP O1G    O         1.740    1.140   -6.670
    HET         0  ATP O2G    O         2.120   -1.040   -7.890
    HET         0  ATP O3G    O        -0.300   -0.140   -7.420
    HET         0  ATP PB     P         0.260   -0.130   -4.450
    HET         0  ATP O1B    O         0.810    1.230   -4.300
    HET         0  ATP O2B    O        -1.230   -0.040   -5.060
    HET         0  ATP O3B    O         1.190   -0.990   -5.430
    HET         0  ATP PA     P        -0.740    0.070   -2.070
    HET         0  ATP O1A    O        -2.100    0.140   -2.670
    HET         0  ATP O2A    O        -0.120    1.550   -1.960
    HET         0  ATP O3A    O         0.200   -0.840   -3.000
    HET         0  ATP HOG2   H         2.100   -0.550   -8.730
    HET         0  ATP HOG3   H        -0.620   -1.050   -7.520
    HET         0  ATP HOB2   H        -1.550   -0.950   -5.130
    HET         0  ATP HOA2   H         0.750    1.460   -1.560
    <BLANKLINE>
    New molecule
    HET         0  ATP O5'    O        -0.840   -0.590   -0.600
    HET         0  ATP C5'    C        -1.690    0.260    0.170
    HET         0  ATP C4'    C        -1.830   -0.310    1.580
    HET         0  ATP O4'    O        -0.540   -0.360    2.230
    HET         0  ATP C3'    C        -2.680    0.630    2.460
    HET         0  ATP O3'    O        -4.030    0.160    2.530
    HET         0  ATP C2'    C        -2.010    0.560    3.860
    HET         0  ATP O2'    O        -2.930    0.040    4.830
    HET         0  ATP C1'    C        -0.830   -0.420    3.650
    HET         0  ATP N9     N         0.330    0.020    4.430
    HET         0  ATP C8     C         1.300    0.880    4.010
    HET         0  ATP N7     N         2.180    1.040    4.960
    HET         0  ATP C5     C         1.830    0.300    6.030
    HET         0  ATP C6     C         2.390    0.080    7.300
    HET         0  ATP N6     N         3.560    0.710    7.680
    HET         0  ATP N1     N         1.760   -0.750    8.140
    HET         0  ATP C2     C         0.640   -1.350    7.780
    HET         0  ATP N3     N         0.090   -1.180    6.600
    HET         0  ATP C4     C         0.640   -0.370    5.700
    HET         0  ATP H5'1   H        -2.680    0.310   -0.300
    HET         0  ATP H5'2   H        -1.260    1.260    0.220
    HET         0  ATP H4'    H        -2.280   -1.300    1.550
    HET         0  ATP H3'    H        -2.650    1.650    2.080
    HET         0  ATP HO3'   H        -4.520    0.790    3.090
    HET         0  ATP H2'    H        -1.650    1.540    4.160
    HET         0  ATP HO2'   H        -3.670    0.660    4.870
    HET         0  ATP H1'    H        -1.120   -1.430    3.930
    HET         0  ATP H8     H         1.330    1.360    3.040
    HET         0  ATP HN61   H         3.940    0.550    8.560
    HET         0  ATP HN62   H         4.020    1.300    7.060
    HET         0  ATP H2     H         0.170   -2.010    8.490
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
